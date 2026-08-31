"""
Survival Pipeline Orchestrator — survival_pipeline.py

Central runner for the Intain-Sight survival analysis engine. Executes:
    1. Panel → Cohort transformation (SurvivalCohortTransformer)
    2. Baseline Kaplan-Meier + Cox PH fitting (BaselineSurvivalEngine)
    3. Competing Risk CIF + Cause-Specific Cox (CompetingRiskEngine)
    4. Portfolio CDR / CPR / Pool Health Score (SecuritizationRateEngine)
    5. Interactive visual artifact generation (SurvivalVisualizer)
    6. C-index benchmarking and MLflow metric logging

All outputs are persisted to the feature store and reports directory.
MLflow experiment: `intain_survival_modeling`
"""

import mlflow
import polars as pl
import pandas as pd
from typing import Dict, List, Optional, Any

from src.survival.data_adapter import SurvivalCohortTransformer
from src.survival.baseline_models import BaselineSurvivalEngine
from src.survival.competing_risks import CompetingRiskEngine
from src.survival.securitization_rates import SecuritizationRateEngine
from src.survival.risk_heatmaps import SurvivalVisualizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Covariates passed to Cox and Competing Risk models
# (Must be numeric or pre-encoded before pipeline invocation)
DEFAULT_COVARIATE_COLS: List[str] = [
    "original_ltv",
    "interest_rate",
    "loan_age",
]


class SurvivalPipeline:
    """
    End-to-end orchestrator for the survival analysis module.

    Designed to be callable from a Jupyter notebook, CLI, or the main
    Intain-Sight inference runner. Logs all artifacts and metrics to MLflow.
    """

    def __init__(
        self,
        rules_path: str = "configs/validation_rules.json",
        mlflow_tracking_uri: str = "sqlite:///mlruns.db",
        experiment_name: str = "intain_survival_modeling",
        covariate_cols: Optional[List[str]] = None,
        penalizer: float = 0.01,
    ):
        """
        Args:
            rules_path: Path to validation_rules.json.
            mlflow_tracking_uri: URI for MLflow tracking.
            experiment_name: MLflow experiment name.
            covariate_cols: List of numeric covariate columns for Cox models.
            penalizer: L2 regularization for Cox models.
        """
        self.covariate_cols = covariate_cols or DEFAULT_COVARIATE_COLS

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.transformer = SurvivalCohortTransformer(rules_path=rules_path)
        self.baseline_engine = BaselineSurvivalEngine(penalizer=penalizer)
        self.competing_engine = CompetingRiskEngine(penalizer=penalizer)
        self.rate_engine = SecuritizationRateEngine()
        self.visualizer = SurvivalVisualizer()

        logger.info(f"SurvivalPipeline initialized. MLflow experiment: '{experiment_name}'")

    def run(
        self,
        panel_df: pl.DataFrame,
        feature_store_path: str = "data/processed/feature_matrix.parquet",
        stratify_km_by: Optional[List[str]] = None,
        segment_rates_by: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full survival analysis pipeline.

        Args:
            panel_df: Monthly loan performance panel (Polars DataFrame).
            feature_store_path: Path to existing Parquet feature store.
            stratify_km_by: Columns to stratify KM curves by.
            segment_rates_by: Columns to segment CDR/CPR by.

        Returns:
            Dict containing all fitted model objects and computed metrics.
        """
        results: Dict[str, Any] = {}

        with mlflow.start_run(run_name="survival_pipeline"):

            # ── Step 1: Panel → Survival Cohort ──────────────────────────────
            logger.info("=== STEP 1: Survival Cohort Transformation ===")
            survival_df = self.transformer.get_survival_dataset(panel_df)
            results["survival_df"] = survival_df

            event_dist = survival_df.group_by("event_status").agg(
                pl.len().alias("count")
            ).sort("event_status").to_dicts()
            for row in event_dist:
                mlflow.log_metric(f"event_{row['event_status']}_count", row["count"])

            # ── Step 2: Baseline Kaplan-Meier + Cox ──────────────────────────
            logger.info("=== STEP 2: Baseline KM & Cox PH Models ===")
            km_stratify = stratify_km_by or ["origination_vintage", "credit_score_band"]

            self.baseline_engine.fit_kaplan_meier(
                survival_df, stratify_by=km_stratify
            ).fit_cox(
                survival_df, covariate_cols=self.covariate_cols, event_col=1
            )

            baseline_cindex = self.baseline_engine.concordance_index()
            mlflow.log_metric("baseline_cox_cindex", baseline_cindex)
            logger.info(f"Baseline Cox C-index: {baseline_cindex:.4f}")
            results["baseline_cox_cindex"] = baseline_cindex

            # ── Step 3: Competing Risk CIF + Cause-Specific Cox ──────────────
            logger.info("=== STEP 3: Competing Risk Models ===")
            self.competing_engine.fit_cif(survival_df).fit_cause_specific_cox(
                survival_df, covariate_cols=self.covariate_cols
            )

            cr_cindices = self.competing_engine.concordance_indices()
            for k, v in cr_cindices.items():
                mlflow.log_metric(k, v)
                logger.info(f"  {k}: {v:.4f}")
            results["competing_risk_cindices"] = cr_cindices

            # Compare baseline vs. competing risk
            cr_default_cindex = cr_cindices.get("cox_default_cindex", 0.0)
            delta = cr_default_cindex - baseline_cindex
            mlflow.log_metric("cindex_improvement_competing_vs_baseline", delta)
            logger.info(f"C-index improvement (Competing vs Baseline): {delta:+.4f}")

            # ── Step 4: Export Competing Risk Features to Feature Store ───────
            logger.info("=== STEP 4: Exporting Competing Risk Meta-Features ===")
            available_covs = [c for c in self.covariate_cols if c in survival_df.columns]
            loan_features_pd = survival_df.select(
                ["loan_id"] + available_covs
            ).to_pandas().set_index("loan_id")

            updated_store = self.competing_engine.export_to_feature_store(
                loan_features=loan_features_pd,
                feature_store_path=feature_store_path,
                time_horizons=[12, 24, 36, 60],
            )
            mlflow.log_param("feature_store_path", updated_store)
            results["updated_feature_store"] = updated_store

            # ── Step 5: CDR / CPR / Pool Health Score ─────────────────────────
            logger.info("=== STEP 5: Securitization Rates (CDR/CPR) ===")
            monthly_rates = self.rate_engine.compute_monthly_rates(panel_df)
            monthly_rates = self.rate_engine.compute_pool_health_score(monthly_rates)

            # Latest month summary
            latest = monthly_rates.sort("reporting_month").tail(1).to_dicts()[0]
            mlflow.log_metrics({
                "latest_cdr": round(float(latest.get("cdr", 0.0)), 6),
                "latest_cpr": round(float(latest.get("cpr", 0.0)), 6),
                "latest_pool_health": round(float(latest.get("pool_health_score", 0.0)), 2),
            })
            results["monthly_rates"] = monthly_rates

            if segment_rates_by:
                segmented_rates = self.rate_engine.compute_segmented_rates(
                    panel_df, segment_cols=segment_rates_by
                )
                results["segmented_rates"] = segmented_rates

            # ── Step 6: Visual Artifacts ──────────────────────────────────────
            logger.info("=== STEP 6: Generating Visual Artifacts ===")

            # CIF plot
            cif_path = self.visualizer.plot_competing_cif(
                cif_default=self.competing_engine.cif_default,
                cif_prepay=self.competing_engine.cif_prepay,
            )
            mlflow.log_artifact(cif_path)

            # CDR/CPR timeseries
            cdr_path = self.visualizer.plot_cdr_cpr_timeseries(monthly_rates)
            mlflow.log_artifact(cdr_path)

            # Hazard heatmap (requires credit_score_band in survival_df)
            if "credit_score_band" in survival_df.columns:
                heatmap_path = self.visualizer.plot_hazard_heatmap(survival_df)
                mlflow.log_artifact(heatmap_path)

            logger.info("=== SURVIVAL PIPELINE COMPLETE ===")
            logger.info(
                f"Baseline Cox C-index:     {baseline_cindex:.4f}\n"
                f"Competing Risk C-index:   {cr_default_cindex:.4f}\n"
                f"Latest CDR:               {latest.get('cdr', 0.0):.4%}\n"
                f"Latest CPR:               {latest.get('cpr', 0.0):.4%}\n"
                f"Latest Pool Health Score: {latest.get('pool_health_score', 0.0):.1f}/100"
            )

        return results
