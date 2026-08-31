"""
Survival Cohort Transformer — data_adapter.py

Converts monthly panel data (one row per loan per month) into loan-level
survival datasets (one row per loan) compatible with lifelines and scikit-survival.

Financial Context:
    In structured finance, each loan has exactly ONE terminal event:
        - Default (90+ DPD / charge-off)      → event_status = 1
        - Prepayment (full unscheduled payoff) → event_status = 2
        - Neither (still active at cutoff)     → event_status = 0 (Censored)

    Treating prepayment as simple censoring (0) inflates default risk estimates —
    the core motivation for competing risk modeling (Fine-Gray / CIF).
"""

import polars as pl
import json
from pathlib import Path
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Baseline static covariates carried forward to the survival frame
BASELINE_COVARIATES: List[str] = [
    "credit_score_band",
    "original_ltv",
    "dti_band",
    "interest_rate",
    "state",
    "origination_vintage",
]


class SurvivalCohortTransformer:
    """
    Transforms monthly loan performance panel data into a loan-level survival dataset.

    Each loan is reduced to a single row containing:
        - duration       : Observed loan age (in months) at event or censoring
        - event_status   : 0 = Censored, 1 = Default, 2 = Prepayment
        - Baseline static covariates at origination
    """

    def __init__(self, rules_path: str = "configs/validation_rules.json"):
        """
        Args:
            rules_path: Path to validation_rules.json for dynamic threshold ingestion.
        """
        rules_file = Path(rules_path)
        if not rules_file.exists():
            raise FileNotFoundError(f"Validation rules not found at: {rules_path}")
        with open(rules_file) as f:
            self._rules = json.load(f)

        self._dpd_default_threshold: int = int(
            self._rules["delinquency_rules"]["dpd_default_threshold"]
        )
        logger.info(
            f"SurvivalCohortTransformer initialized. "
            f"Default DPD threshold: {self._dpd_default_threshold}"
        )

    def transform(self, panel_df: pl.DataFrame) -> pl.DataFrame:
        """
        Aggregates monthly panel records into one survival row per loan.

        Priority for event assignment (mutually exclusive, first-wins):
            1. Default  — any record with days_past_due >= threshold OR default_flag == 1
            2. Prepayment — any record with prepayment_flag == 1
            3. Censored — all other active loans

        Args:
            panel_df: Monthly loan performance DataFrame (Polars).
                      Required columns: loan_id, loan_age, days_past_due,
                      default_flag, prepayment_flag, + BASELINE_COVARIATES.

        Returns:
            pl.DataFrame: Loan-level survival frame with columns:
                loan_id, duration, event_status, <baseline_covariates>
        """
        logger.info(f"Transforming {panel_df.height:,} panel rows into survival cohorts...")
        self._validate_columns(panel_df)

        # Build per-loan event flags: did the loan EVER hit each terminal state?
        default_flag_col = (
            (pl.col("days_past_due") >= self._dpd_default_threshold) |
            (pl.col("default_flag") == 1)
        ).cast(pl.Int32)

        prepay_flag_col = (pl.col("prepayment_flag") == 1).cast(pl.Int32)

        panel_df = panel_df.with_columns([
            default_flag_col.alias("_is_default"),
            prepay_flag_col.alias("_is_prepay"),
        ])

        # Aggregate to loan-level
        loan_agg = panel_df.group_by("loan_id").agg([
            pl.col("loan_age").max().alias("duration"),
            pl.col("_is_default").max().alias("_ever_default"),
            pl.col("_is_prepay").max().alias("_ever_prepay"),
            # Take the static covariate from the earliest available record
            *[pl.col(c).first().alias(c) for c in BASELINE_COVARIATES if c in panel_df.columns],
        ])

        # Assign event_status: Default (1) > Prepayment (2) > Censored (0)
        loan_agg = loan_agg.with_columns(
            pl.when(pl.col("_ever_default") == 1)
            .then(pl.lit(1))
            .when(pl.col("_ever_prepay") == 1)
            .then(pl.lit(2))
            .otherwise(pl.lit(0))
            .cast(pl.Int32)
            .alias("event_status")
        ).drop(["_ever_default", "_ever_prepay"])

        # Filter invalid cohorts: duration must be >= 1 month
        invalid = loan_agg.filter(pl.col("duration") < 1).height
        if invalid > 0:
            logger.warning(f"Dropping {invalid} loans with duration < 1 month.")
        loan_agg = loan_agg.filter(pl.col("duration") >= 1)

        event_counts = loan_agg.group_by("event_status").agg(pl.len().alias("count")).sort("event_status")
        logger.info(f"Survival cohort built. Event distribution:\n{event_counts}")

        return loan_agg

    def get_survival_dataset(self, panel_df: pl.DataFrame) -> pl.DataFrame:
        """
        Public entry point — alias for transform() for API consistency.

        Args:
            panel_df: Monthly loan performance panel.

        Returns:
            pl.DataFrame: Clean loan-level survival cohort.
        """
        return self.transform(panel_df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pl.DataFrame) -> None:
        """Asserts required columns exist before processing."""
        required = {"loan_id", "loan_age", "days_past_due", "default_flag", "prepayment_flag"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Panel DataFrame missing required columns: {missing}")
