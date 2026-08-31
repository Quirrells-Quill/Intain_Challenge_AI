"""
Anomaly Pipeline Orchestrator — anomaly_pipeline.py

Central runner for the Intain-Sight Hybrid Anomaly & Exception Detection Engine.
Executes the full 5-tier pipeline and emits all 6 anomaly-related columns
required by the submission_template.csv schema (PILLAR 1 compliance).

Output columns:
    loan_id, reporting_month, exception_required, exception_type,
    anomaly_score, top_drivers, recommended_action

Additionally writes:
    data/processed/test_anomaly_predictions.parquet  — cached ML scoring outputs
    reports/ANOMALY_REVIEWER_DOSSIER.md              — 20-case reviewer dossier
    reports/ANOMALY_REVIEWER_DOSSIER.html            — HTML sign-off table
"""

import polars as pl
import numpy as np
import mlflow
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.anomaly.rule_detector import DeterministicRuleAuditor
from src.anomaly.reconciliation import ServicerReconciliationEngine
from src.anomaly.ml_detectors import UnsupervisedAnomalyStack
from src.anomaly.score_fusion import AnomalyFusionEngine
from src.anomaly.explainer import AnomalyExplainer
from src.anomaly.report_generator import AnomalyAuditReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Submission columns produced by this pipeline (subset of full 13-column schema)
ANOMALY_OUTPUT_COLS: List[str] = [
    "loan_id",
    "reporting_month",
    "exception_required",
    "exception_type",
    "anomaly_score",
    "top_drivers",
    "recommended_action",
    "confidence",
]


class AnomalyPipeline:
    """
    Orchestrates the complete Hybrid Anomaly Detection pipeline.

    Execution order:
        1. DeterministicRuleAuditor   → Rule violation flags + rule driver codes
        2. ServicerReconciliationEngine → Servicer discrepancy flags + notes
        3. UnsupervisedAnomalyStack   → Isolation Forest + Autoencoder scores
        4. AnomalyFusionEngine        → Composite 0–100 score + exception columns
        5. AnomalyExplainer           → top_drivers strings (SHAP + rule codes)
        6. AnomalyAuditReportGenerator → 20-case reviewer dossier (MD + HTML)
    """

    def __init__(
        self,
        rules_path: str = "configs/validation_rules.json",
        mlflow_tracking_uri: str = "sqlite:///mlruns.db",
        experiment_name: str = "intain_anomaly_detection",
        ae_epochs: int = 30,
        iso_contamination: float = 0.03,
    ):
        """
        Args:
            rules_path: Path to validation_rules.json.
            mlflow_tracking_uri: MLflow tracking server URI.
            experiment_name: Name of the MLflow experiment.
            ae_epochs: Autoencoder training epochs.
            iso_contamination: Expected outlier fraction for Isolation Forest.
        """
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.rule_auditor = DeterministicRuleAuditor(rules_path=rules_path)
        self.reconciliation = ServicerReconciliationEngine(rules_path=rules_path)
        self.ml_stack = UnsupervisedAnomalyStack(
            contamination=iso_contamination,
            ae_epochs=ae_epochs,
        )
        self.fusion_engine = AnomalyFusionEngine()
        self.report_generator = AnomalyAuditReportGenerator()

        logger.info(f"AnomalyPipeline initialized. Experiment: '{experiment_name}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        test_df: pl.DataFrame,
        train_df: Optional[pl.DataFrame] = None,
        servicer_df: Optional[pl.DataFrame] = None,
        output_parquet: str = "data/processed/test_anomaly_predictions.parquet",
    ) -> pl.DataFrame:
        """
        Executes the full anomaly detection pipeline on the test dataset.

        Args:
            test_df: Monthly loan performance test DataFrame (Polars).
            train_df: Optional training DataFrame used to fit ML models.
                      If None, ML models are fitted on test_df (unsupervised).
            servicer_df: Optional servicer update feed for reconciliation.
                         If None, reconciliation step is skipped.
            output_parquet: Path to write cached anomaly predictions.

        Returns:
            pl.DataFrame: Test records with all 8 anomaly output columns.
                          Guaranteed zero nulls across ANOMALY_OUTPUT_COLS.
        """
        with mlflow.start_run(run_name="anomaly_pipeline"):

            # ── Step 1: Deterministic Rule Auditing ───────────────────────
            logger.info("=== STEP 1: Deterministic Rule Auditing ===")
            df = self.rule_auditor.audit(test_df)
            rule_drivers = self.rule_auditor.get_top_drivers(df)
            df = df.with_columns(rule_drivers)

            n_rule_violations = df.filter(pl.col("rule_violation_count") > 0).height
            mlflow.log_metric("rule_violations", n_rule_violations)
            logger.info(f"Rule violations: {n_rule_violations:,}")

            # ── Step 2: Servicer Reconciliation ──────────────────────────
            logger.info("=== STEP 2: Servicer Reconciliation ===")
            if servicer_df is not None:
                df = self.reconciliation.reconcile(df, servicer_df)
            else:
                logger.warning("No servicer_df provided — skipping reconciliation.")
                df = df.with_columns([
                    pl.lit(False).alias("servicer_discrepancy_flag"),
                    pl.lit("Reconciliation skipped — no servicer feed provided.").alias("reconciliation_notes"),
                ])

            n_svc_conflicts = df["servicer_discrepancy_flag"].sum()
            mlflow.log_metric("servicer_discrepancies", n_svc_conflicts)
            logger.info(f"Servicer discrepancies: {n_svc_conflicts:,}")

            # ── Step 3: ML Unsupervised Anomaly Scoring ───────────────────
            logger.info("=== STEP 3: Unsupervised ML Anomaly Scoring ===")
            fit_df = train_df if train_df is not None else df
            X_fit, feature_names = UnsupervisedAnomalyStack.select_numeric_features(fit_df)
            X_test, _ = UnsupervisedAnomalyStack.select_numeric_features(
                df.select([c for c in feature_names if c in df.columns])
            )

            logger.info(f"ML feature matrix: {X_fit.shape[0]:,} train rows × {X_fit.shape[1]} features")
            self.ml_stack.fit(X_fit)
            iso_scores, ae_scores = self.ml_stack.score(X_test)

            df = df.with_columns([
                pl.Series("iso_anomaly_score", iso_scores.astype(float)),
                pl.Series("ae_anomaly_score", ae_scores.astype(float)),
            ])

            mlflow.log_metric("mean_iso_score", float(iso_scores.mean()))
            mlflow.log_metric("mean_ae_score", float(ae_scores.mean()))

            # ── Step 4: Score Fusion ───────────────────────────────────────
            logger.info("=== STEP 4: Anomaly Score Fusion ===")
            df = self.fusion_engine.fuse(
                df,
                rule_violation_col="rule_violation_count",
                servicer_conflict_col="servicer_discrepancy_flag",
                iso_score_col="iso_anomaly_score",
                ae_score_col="ae_anomaly_score",
            )

            mlflow.log_metric("mean_anomaly_score", float(df["anomaly_score"].mean()))
            mlflow.log_metric(
                "exception_rate",
                float(df["exception_required"].sum() / max(df.height, 1))
            )

            # ── Step 5: SHAP Explanation & top_drivers ────────────────────
            logger.info("=== STEP 5: Anomaly Explanation (SHAP + Rule Codes) ===")
            explainer = AnomalyExplainer(
                anomaly_stack=self.ml_stack,
                feature_names=feature_names,
            )
            try:
                X_scaled = self.ml_stack.scaler.transform(X_fit)
                explainer.build_shap_explainer(X_scaled)
                X_test_scaled = self.ml_stack.scaler.transform(X_test)
                top_drivers = explainer.explain_batch(df, X_test_scaled)
            except Exception as e:
                logger.warning(f"SHAP explanation failed ({e}). Using rule drivers as fallback.")
                top_drivers = df["rule_drivers"] if "rule_drivers" in df.columns else pl.Series("top_drivers", ["None"] * df.height)

            df = df.with_columns(top_drivers.alias("top_drivers"))

            # ── Step 6: Anomaly Audit Dossier ─────────────────────────────
            logger.info("=== STEP 6: Generating Anomaly Reviewer Dossier ===")
            dossier_path = self.report_generator.generate_report(df)
            mlflow.log_artifact(dossier_path)
            logger.info(f"Dossier logged to MLflow: {dossier_path}")

            # ── Schema Validation & Export ────────────────────────────────
            logger.info("=== STEP 7: Schema Validation & Parquet Export ===")
            output_df = self._build_output(df)
            self._validate_zero_nulls(output_df)

            out_path = Path(output_parquet)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.write_parquet(str(out_path), compression="snappy")
            mlflow.log_param("output_parquet", str(out_path))
            logger.info(f"Anomaly predictions cached → {out_path} ({output_df.height:,} rows)")

            logger.info("=== ANOMALY PIPELINE COMPLETE ===")

        return output_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_output(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Selects and type-casts the 8 anomaly output columns.
        Fills any remaining nulls with safe defaults to guarantee zero nulls.
        """
        out = df.with_columns([
            pl.col("exception_required").fill_null(False),
            pl.col("exception_type").fill_null("None"),
            pl.col("anomaly_score").fill_null(0.0).cast(pl.Float64),
            pl.col("top_drivers").fill_null("None"),
            pl.col("recommended_action").fill_null("Auto-Approve"),
            pl.col("confidence").fill_null(1.0).cast(pl.Float64),
        ])

        available = [c for c in ANOMALY_OUTPUT_COLS if c in out.columns]
        return out.select(available)

    def _validate_zero_nulls(self, df: pl.DataFrame) -> None:
        """Raises ValueError if any output column contains nulls (PILLAR 1)."""
        for col in df.columns:
            n_nulls = df[col].null_count()
            if n_nulls > 0:
                raise ValueError(
                    f"PILLAR 1 VIOLATION — Column '{col}' has {n_nulls} nulls in anomaly output."
                )
        logger.info("PILLAR 1 schema validation PASSED — zero nulls confirmed.")
