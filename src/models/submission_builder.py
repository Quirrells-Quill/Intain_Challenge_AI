"""
Submission Builder — Schema Contract Enforcer (PILLAR 1)

Validates and constructs the final submission DataFrame against the
13-column schema mandated by the Intain-Sight evaluation harness.
Any deviation results in automated disqualification.
"""

import polars as pl
import json
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Immutable schema contract (mirrors submission_template.csv exactly)
# ---------------------------------------------------------------------------
SUBMISSION_SCHEMA = {
    "loan_id":               pl.Utf8,
    "reporting_month":       pl.Date,
    "prob_3m_delinq":        pl.Float64,
    "prob_6m_delinq":        pl.Float64,
    "prob_12m_default":      pl.Float64,
    "prob_12m_prepay":       pl.Float64,
    "predicted_next_state":  pl.Utf8,
    "exception_required":    pl.Boolean,
    "exception_type":        pl.Utf8,
    "anomaly_score":         pl.Float64,
    "top_drivers":           pl.Utf8,
    "recommended_action":    pl.Utf8,
    "confidence":            pl.Float64,
}


class SubmissionBuilder:
    """
    Constructs, validates, and exports the final submission DataFrame.

    Ingests all thresholds dynamically from configs/validation_rules.json —
    no hardcoded numeric boundaries anywhere in this class (PILLAR 2 compliance).
    """

    def __init__(self, rules_path: str = "configs/validation_rules.json"):
        """
        Args:
            rules_path: Path to the validation rules JSON (dynamically loaded).
        """
        rules_file = Path(rules_path)
        if not rules_file.exists():
            raise FileNotFoundError(f"Validation rules not found: {rules_path}")

        with open(rules_file) as f:
            self._rules = json.load(f)

        self._bounds = self._rules["submission_bounds"]
        logger.info(f"SubmissionBuilder initialized with rules from {rules_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        loan_ids: pl.Series,
        reporting_months: pl.Series,
        prob_3m_delinq: pl.Series,
        prob_6m_delinq: pl.Series,
        prob_12m_default: pl.Series,
        prob_12m_prepay: pl.Series,
        predicted_next_state: pl.Series,
        anomaly_scores: pl.Series,
        top_drivers: pl.Series,
        confidence: pl.Series,
        exception_score_threshold: Optional[float] = None,
    ) -> pl.DataFrame:
        """
        Assembles the 13-column submission DataFrame and validates all bounds.

        The exception_score_threshold is loaded from validation_rules.json if
        not explicitly overridden — enforcing PILLAR 2's no-hardcoding contract.

        Args:
            loan_ids: Series of loan identifiers.
            reporting_months: Series of reporting dates.
            prob_3m_delinq: Calibrated 3-month delinquency probabilities.
            prob_6m_delinq: Calibrated 6-month delinquency probabilities.
            prob_12m_default: Calibrated 12-month default probabilities.
            prob_12m_prepay: Calibrated 12-month prepayment probabilities.
            predicted_next_state: Multiclass next-state predictions.
            anomaly_scores: Normalized anomaly scores [0, 100].
            top_drivers: Semicolon-delimited SHAP/rule driver strings.
            confidence: Ensemble variance-derived confidence scores.
            exception_score_threshold: Override threshold for exception_required.

        Returns:
            pl.DataFrame: Validated 13-column submission frame.

        Raises:
            ValueError: If any column fails schema or bounds validation.
        """
        # Load threshold from rules if not overridden
        cutoff = (
            exception_score_threshold
            if exception_score_threshold is not None
            else float(self._rules["anomaly_thresholds"]["anomaly_score_exception_cutoff"])
        )

        # Derive exception_required and exception_type from anomaly scores
        exception_required = anomaly_scores > cutoff
        exception_type = self._derive_exception_type(
            anomaly_scores=anomaly_scores,
            exception_required=exception_required,
        )

        # Derive recommended_action from probabilities and exception status
        recommended_action = self._derive_recommended_action(
            prob_12m_default=prob_12m_default,
            exception_required=exception_required,
        )

        df = pl.DataFrame({
            "loan_id":              loan_ids.cast(pl.Utf8),
            "reporting_month":      reporting_months.cast(pl.Date),
            "prob_3m_delinq":       prob_3m_delinq.cast(pl.Float64),
            "prob_6m_delinq":       prob_6m_delinq.cast(pl.Float64),
            "prob_12m_default":     prob_12m_default.cast(pl.Float64),
            "prob_12m_prepay":      prob_12m_prepay.cast(pl.Float64),
            "predicted_next_state": predicted_next_state.cast(pl.Utf8),
            "exception_required":   exception_required,
            "exception_type":       exception_type,
            "anomaly_score":        anomaly_scores.cast(pl.Float64),
            "top_drivers":          top_drivers.cast(pl.Utf8),
            "recommended_action":   recommended_action,
            "confidence":           confidence.cast(pl.Float64),
        })

        self._validate(df)
        logger.info(f"SubmissionBuilder: valid submission frame built ({df.height:,} rows).")
        return df

    def export(self, df: pl.DataFrame, output_path: str = "reports/submission.csv") -> str:
        """
        Exports the validated submission frame to CSV, preserving row order.

        Args:
            df: Validated submission DataFrame from build().
            output_path: Destination CSV path.

        Returns:
            Absolute path to the written file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(str(out))
        logger.info(f"Submission CSV written to {out} ({df.height:,} rows).")
        return str(out)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_exception_type(
        self,
        anomaly_scores: pl.Series,
        exception_required: pl.Series,
    ) -> pl.Series:
        """
        Maps anomaly scores to exception type strings dynamically using rule thresholds.
        Thresholds are loaded from validation_rules.json (PILLAR 2).
        """
        severe_cutoff = float(
            self._rules["anomaly_thresholds"]["severe_deterioration_score"]
        )
        normal_cutoff = float(
            self._rules["anomaly_thresholds"]["anomaly_score_exception_cutoff"]
        )

        return (
            pl.when(~exception_required)
            .then(pl.lit("None"))
            .when(anomaly_scores >= severe_cutoff)
            .then(pl.lit("Severe Deterioration"))
            .when(anomaly_scores >= normal_cutoff)
            .then(pl.lit("Servicer Discrepancy"))
            .otherwise(pl.lit("Data Logic Error"))
            .alias("exception_type")
        )

    def _derive_recommended_action(
        self,
        prob_12m_default: pl.Series,
        exception_required: pl.Series,
    ) -> pl.Series:
        """
        Maps model output to recommended actions using rule-loaded thresholds.
        """
        reject_threshold = float(self._rules["anomaly_thresholds"]["severe_deterioration_score"]) / 100.0
        triage_threshold = float(self._rules["anomaly_thresholds"]["anomaly_score_exception_cutoff"]) / 100.0

        return (
            pl.when(prob_12m_default >= reject_threshold)
            .then(pl.lit("Reject/Repurchase"))
            .when(exception_required | (prob_12m_default >= triage_threshold))
            .then(pl.lit("Manual Triage"))
            .otherwise(pl.lit("Auto-Approve"))
            .alias("recommended_action")
        )

    def _validate(self, df: pl.DataFrame) -> None:
        """
        Enforces PILLAR 1 schema contract: column presence, types, and value bounds.

        Raises:
            ValueError: On any schema or bounds violation.
        """
        # 1. Column completeness
        missing_cols = [c for c in SUBMISSION_SCHEMA if c not in df.columns]
        if missing_cols:
            raise ValueError(f"PILLAR 1 VIOLATION — Missing submission columns: {missing_cols}")

        # 2. Zero null enforcement
        null_counts = {c: df[c].null_count() for c in SUBMISSION_SCHEMA if df[c].null_count() > 0}
        if null_counts:
            raise ValueError(f"PILLAR 1 VIOLATION — Null values detected: {null_counts}")

        # 3. Probability bounds [0.0, 1.0]
        p_min = float(self._bounds["probability_min"])
        p_max = float(self._bounds["probability_max"])
        for col in ["prob_3m_delinq", "prob_6m_delinq", "prob_12m_default", "prob_12m_prepay", "confidence"]:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < p_min or col_max > p_max:
                raise ValueError(
                    f"PILLAR 1 VIOLATION — '{col}' out of bounds [{p_min}, {p_max}]: "
                    f"min={col_min}, max={col_max}"
                )

        # 4. Anomaly score bounds [0.0, 100.0]
        a_min = float(self._bounds["anomaly_score_min"])
        a_max = float(self._bounds["anomaly_score_max"])
        if df["anomaly_score"].min() < a_min or df["anomaly_score"].max() > a_max:
            raise ValueError(f"PILLAR 1 VIOLATION — 'anomaly_score' out of bounds [{a_min}, {a_max}]")

        # 5. Categorical value validation
        valid_states = set(self._bounds["valid_next_states"])
        invalid_states = set(df["predicted_next_state"].unique().to_list()) - valid_states
        if invalid_states:
            raise ValueError(f"PILLAR 1 VIOLATION — Invalid predicted_next_state values: {invalid_states}")

        valid_ex_types = set(self._bounds["valid_exception_types"])
        invalid_ex = set(df["exception_type"].unique().to_list()) - valid_ex_types
        if invalid_ex:
            raise ValueError(f"PILLAR 1 VIOLATION — Invalid exception_type values: {invalid_ex}")

        valid_actions = set(self._bounds["valid_actions"])
        invalid_actions = set(df["recommended_action"].unique().to_list()) - valid_actions
        if invalid_actions:
            raise ValueError(f"PILLAR 1 VIOLATION — Invalid recommended_action values: {invalid_actions}")

        logger.info("PILLAR 1 schema validation PASSED — submission frame is compliant.")
