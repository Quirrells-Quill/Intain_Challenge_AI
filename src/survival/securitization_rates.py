"""
Securitization Rate Engine — securitization_rates.py

Computes industry-standard structured finance prepayment and default velocity
metrics from monthly loan-level panel data.

Financial Definitions:
    MDR (Monthly Default Rate):
        Fraction of the beginning principal balance that defaulted in a given month.
        MDR_t = Defaulted_Balance_t / Beginning_Balance_t

    CDR (Conditional Default Rate):
        MDR annualized via compounding convention used in ABS/MBS analytics.
        CDR_t = 1 - (1 - MDR_t)^12

    SMM (Single Monthly Mortality):
        Fraction of the *schedulable* remaining balance that was prepaid in a month.
        SMM_t = Unscheduled_Principal_t / (Beginning_Balance_t - Scheduled_Principal_t)

    CPR (Conditional Prepayment Rate):
        SMM annualized — the standard Wall Street prepayment speed metric.
        CPR_t = 1 - (1 - SMM_t)^12

    These rates underpin Bloomberg ABS pricing, CDO waterfall modeling, and
    servicer performance benchmarking in Intain's pool verification system.
"""

import polars as pl
import numpy as np
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SecuritizationRateEngine:
    """
    Computes monthly MDR→CDR and SMM→CPR time series for an asset pool.

    Supports segment-level slicing by vintage and servicer to identify
    geographic or servicer-specific performance outliers.
    """

    def __init__(self):
        logger.info("SecuritizationRateEngine initialized.")

    # ------------------------------------------------------------------
    # Portfolio-Level CDR / CPR
    # ------------------------------------------------------------------

    def compute_monthly_rates(self, panel_df: pl.DataFrame) -> pl.DataFrame:
        """
        Computes MDR, CDR, SMM, and CPR for every reporting month in the pool.

        Required columns:
            reporting_month, current_balance, default_flag,
            scheduled_principal, prepayment_flag, prepay_amount

        Args:
            panel_df: Monthly loan-level performance panel (Polars DataFrame).

        Returns:
            pl.DataFrame: Monthly time series with columns:
                reporting_month, beginning_balance, defaulted_balance,
                unscheduled_prepay, mdr, cdr, smm, cpr
        """
        logger.info("Computing portfolio-level MDR/CDR/SMM/CPR time series...")

        self._validate_columns(panel_df)

        # Compute per-loan beginning balance as prior month's current_balance
        panel_sorted = panel_df.sort(["loan_id", "reporting_month"])
        panel_sorted = panel_sorted.with_columns(
            pl.col("current_balance")
            .shift(1)
            .over("loan_id")
            .alias("beginning_balance")
        ).filter(pl.col("beginning_balance").is_not_null())

        # Monthly aggregation to portfolio level
        monthly = panel_sorted.group_by("reporting_month").agg([
            pl.col("beginning_balance").sum().alias("beginning_balance"),
            (pl.col("default_flag") * pl.col("beginning_balance")).sum().alias("defaulted_balance"),
            pl.col("prepay_amount").sum().alias("unscheduled_prepay"),
            pl.col("scheduled_principal").sum().alias("scheduled_principal"),
        ]).sort("reporting_month")

        # MDR → CDR
        monthly = monthly.with_columns([
            (pl.col("defaulted_balance") / pl.col("beginning_balance").clip(lower_bound=1.0))
            .alias("mdr"),
        ])
        monthly = monthly.with_columns([
            (1.0 - (1.0 - pl.col("mdr")).pow(12)).alias("cdr"),
        ])

        # SMM → CPR
        denominator = (pl.col("beginning_balance") - pl.col("scheduled_principal")).clip(lower_bound=1.0)
        monthly = monthly.with_columns([
            (pl.col("unscheduled_prepay") / denominator).alias("smm"),
        ])
        monthly = monthly.with_columns([
            (1.0 - (1.0 - pl.col("smm")).pow(12)).alias("cpr"),
        ])

        logger.info(f"CDR/CPR computed for {monthly.height} months.")
        return monthly

    # ------------------------------------------------------------------
    # Segmented CDR / CPR
    # ------------------------------------------------------------------

    def compute_segmented_rates(
        self,
        panel_df: pl.DataFrame,
        segment_cols: List[str],
    ) -> pl.DataFrame:
        """
        Computes CDR and CPR grouped by segment columns (vintage, servicer, state).

        This surfaces servicer-specific prepayment anomalies or vintage-driven
        default clustering — critical inputs for the Verification Agent's
        exception_type classification.

        Args:
            panel_df: Monthly loan-level performance panel.
            segment_cols: Columns to segment by (e.g., ['origination_vintage', 'servicer_name']).

        Returns:
            pl.DataFrame: CDR/CPR per (reporting_month, segment) combination.
        """
        logger.info(f"Computing segmented CDR/CPR by: {segment_cols}")

        self._validate_columns(panel_df)

        panel_sorted = panel_df.sort(["loan_id", "reporting_month"])
        panel_sorted = panel_sorted.with_columns(
            pl.col("current_balance")
            .shift(1)
            .over("loan_id")
            .alias("beginning_balance")
        ).filter(pl.col("beginning_balance").is_not_null())

        group_keys = ["reporting_month"] + [c for c in segment_cols if c in panel_sorted.columns]

        segmented = panel_sorted.group_by(group_keys).agg([
            pl.col("beginning_balance").sum().alias("beginning_balance"),
            (pl.col("default_flag") * pl.col("beginning_balance")).sum().alias("defaulted_balance"),
            pl.col("prepay_amount").sum().alias("unscheduled_prepay"),
            pl.col("scheduled_principal").sum().alias("scheduled_principal"),
        ]).sort(group_keys)

        # MDR → CDR
        segmented = segmented.with_columns(
            (pl.col("defaulted_balance") / pl.col("beginning_balance").clip(lower_bound=1.0)).alias("mdr")
        )
        segmented = segmented.with_columns(
            (1.0 - (1.0 - pl.col("mdr")).pow(12)).alias("cdr")
        )

        # SMM → CPR
        denom = (pl.col("beginning_balance") - pl.col("scheduled_principal")).clip(lower_bound=1.0)
        segmented = segmented.with_columns(
            (pl.col("unscheduled_prepay") / denom).alias("smm")
        )
        segmented = segmented.with_columns(
            (1.0 - (1.0 - pl.col("smm")).pow(12)).alias("cpr")
        )

        logger.info(f"Segmented rates computed: {segmented.height} rows.")
        return segmented

    # ------------------------------------------------------------------
    # Pool Health Score  (PILLAR 5)
    # ------------------------------------------------------------------

    def compute_pool_health_score(self, monthly_rates: pl.DataFrame) -> pl.DataFrame:
        """
        Computes a composite Pool Health Score (0–100) for each reporting month.

        Methodology:
            CDR component  : Higher CDR → lower score. Max penalized at CDR=0.10 (10% ann.)
            CPR component  : Moderate CPR is healthy. Extreme CPR (>0.30) penalizes WAM stability.
            Combined score : 100 × (1 - cdr_penalty) × (1 - cpr_penalty)

        Args:
            monthly_rates: Output of compute_monthly_rates().

        Returns:
            pl.DataFrame: monthly_rates with appended pool_health_score column.
        """
        cdr_penalty = (pl.col("cdr") / 0.10).clip(0.0, 1.0)
        cpr_excess  = ((pl.col("cpr") - 0.30) / 0.30).clip(0.0, 1.0)

        return monthly_rates.with_columns(
            (100.0 * (1.0 - cdr_penalty) * (1.0 - cpr_excess * 0.5))
            .round(2)
            .alias("pool_health_score")
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pl.DataFrame) -> None:
        required = {
            "loan_id", "reporting_month", "current_balance",
            "default_flag", "prepay_amount", "scheduled_principal",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Panel DataFrame missing columns for rate computation: {missing}")
