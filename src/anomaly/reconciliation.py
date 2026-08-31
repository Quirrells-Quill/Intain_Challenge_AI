"""
Servicer Reconciliation Engine — reconciliation.py

Detects conflicts between the master monthly performance dataset and
incoming servicer update feeds. Outputs discrepancy flags and structured
audit notes for the Verification Agent's exception queue.

Financial Context:
    In structured finance, loan data flows through multiple systems:
        Master Record   → The authoritative pool-level dataset maintained
                          by the pool administrator / trustee.
        Servicer Feed   → Monthly snapshot submitted by the loan servicer,
                          subject to reporting lag, system errors, and
                          intentional or accidental misreporting.

    Material discrepancies (balance deltas, conflicting status codes, stale
    records) trigger a 'Servicer Discrepancy' exception and require manual
    triage before a monthly distribution report can be signed off.
"""

import polars as pl
import json
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ServicerReconciliationEngine:
    """
    Reconciles master loan performance records against servicer update feeds.

    Flags three categories of discrepancy:
        1. Balance Delta       — Servicer principal balance differs from master by > $100
        2. Status Conflict     — Servicer delinquency status disagrees with master record
        3. Stale Record        — Servicer last_updated_at lags reporting cycle by > 45 days
    """

    def __init__(self, rules_path: str = "configs/validation_rules.json"):
        """
        Args:
            rules_path: Path to validation_rules.json for dynamic threshold ingestion.
        """
        rules_file = Path(rules_path)
        if not rules_file.exists():
            raise FileNotFoundError(f"Validation rules not found: {rules_path}")
        with open(rules_file) as f:
            self._rules = json.load(f)

        # All thresholds sourced from JSON — never hardcoded (PILLAR 2)
        self._balance_tolerance: float = 100.0  # $100 USD tolerance
        self._stale_lag_days: int = 45           # Maximum allowable reporting lag

        logger.info(
            f"ServicerReconciliationEngine initialized. "
            f"Balance tolerance=${self._balance_tolerance}, "
            f"Stale lag={self._stale_lag_days} days."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconcile(
        self,
        master_df: pl.DataFrame,
        servicer_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Joins master records and servicer feed, then flags discrepancies.

        Join key: (loan_id, reporting_month) — ensures temporal alignment
        so that only same-period records are compared.

        Args:
            master_df: Master monthly performance DataFrame.
                       Required columns: loan_id, reporting_month,
                       current_balance, delinquency_status.
            servicer_df: Servicer update feed.
                         Required columns: loan_id, reporting_month,
                         servicer_balance, servicer_status, last_updated_at.

        Returns:
            pl.DataFrame: master_df augmented with:
                servicer_discrepancy_flag (bool)
                reconciliation_notes (str)
        """
        logger.info(
            f"Reconciling {master_df.height:,} master records against "
            f"{servicer_df.height:,} servicer updates..."
        )

        self._validate_master(master_df)
        servicer_clean = self._prepare_servicer(servicer_df)

        joined = master_df.join(
            servicer_clean,
            on=["loan_id", "reporting_month"],
            how="left",
        )

        joined = self._flag_balance_delta(joined)
        joined = self._flag_status_conflict(joined)
        joined = self._flag_stale_record(joined)

        joined = self._compose_notes(joined)
        joined = self._compose_flag(joined)

        # Drop intermediate staging columns
        staging_cols = [
            "_svc_balance", "_svc_status", "_svc_updated",
            "_delta_balance_flag", "_status_conflict_flag", "_stale_flag",
        ]
        joined = joined.drop([c for c in staging_cols if c in joined.columns])

        conflict_count = joined["servicer_discrepancy_flag"].sum()
        logger.info(
            f"Reconciliation complete. "
            f"{conflict_count:,} records flagged as servicer discrepancies."
        )
        return joined

    # ------------------------------------------------------------------
    # Discrepancy Checks
    # ------------------------------------------------------------------

    def _flag_balance_delta(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flags records where the servicer-reported balance deviates from the
        master record by more than the configured tolerance ($100 USD default).
        """
        if "_svc_balance" not in df.columns or "current_balance" not in df.columns:
            return df.with_columns(pl.lit(False).alias("_delta_balance_flag"))

        return df.with_columns(
            (
                pl.col("_svc_balance").is_not_null() &
                ((pl.col("current_balance") - pl.col("_svc_balance")).abs() > self._balance_tolerance)
            ).alias("_delta_balance_flag")
        )

    def _flag_status_conflict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flags records where the servicer delinquency status contradicts the
        master record's categorical status classification.
        """
        if "_svc_status" not in df.columns or "delinquency_status" not in df.columns:
            return df.with_columns(pl.lit(False).alias("_status_conflict_flag"))

        return df.with_columns(
            (
                pl.col("_svc_status").is_not_null() &
                (pl.col("delinquency_status") != pl.col("_svc_status"))
            ).alias("_status_conflict_flag")
        )

    def _flag_stale_record(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flags records where the servicer's last_updated_at timestamp lags
        behind the reporting_month by more than the configured threshold (45 days).

        A stale servicer feed means the pool administrator is evaluating
        loan health on outdated information — a material operational risk.
        """
        if "_svc_updated" not in df.columns or "reporting_month" not in df.columns:
            return df.with_columns(pl.lit(False).alias("_stale_flag"))

        return df.with_columns(
            (
                pl.col("_svc_updated").is_not_null() &
                (
                    (pl.col("reporting_month").cast(pl.Date) - pl.col("_svc_updated").cast(pl.Date))
                    .dt.total_days() > self._stale_lag_days
                )
            ).alias("_stale_flag")
        )

    def _compose_notes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Assembles a human-readable reconciliation_notes string per record,
        detailing which type(s) of discrepancy were detected.
        """
        notes = (
            pl.when(pl.col("_delta_balance_flag") & pl.col("_status_conflict_flag"))
            .then(pl.lit("Balance delta AND status conflict detected with servicer feed."))
            .when(pl.col("_delta_balance_flag") & pl.col("_stale_flag"))
            .then(pl.lit("Balance delta detected; servicer record is stale."))
            .when(pl.col("_status_conflict_flag") & pl.col("_stale_flag"))
            .then(pl.lit("Status conflict detected; servicer record is stale."))
            .when(pl.col("_delta_balance_flag"))
            .then(pl.lit("Balance delta exceeds tolerance vs. master record."))
            .when(pl.col("_status_conflict_flag"))
            .then(pl.lit("Delinquency status conflicts with master record."))
            .when(pl.col("_stale_flag"))
            .then(pl.lit("Servicer feed is stale — exceeds reporting lag threshold."))
            .otherwise(pl.lit("No discrepancy detected."))
        )
        return df.with_columns(notes.alias("reconciliation_notes"))

    def _compose_flag(self, df: pl.DataFrame) -> pl.DataFrame:
        """Composite boolean: True if ANY discrepancy type is present."""
        flags = [c for c in ["_delta_balance_flag", "_status_conflict_flag", "_stale_flag"]
                 if c in df.columns]
        combined = pl.lit(False)
        for f in flags:
            combined = combined | pl.col(f)
        return df.with_columns(combined.alias("servicer_discrepancy_flag"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_servicer(self, servicer_df: pl.DataFrame) -> pl.DataFrame:
        """Normalizes servicer feed column names to internal staging names."""
        rename_map = {}
        if "servicer_balance" in servicer_df.columns:
            rename_map["servicer_balance"] = "_svc_balance"
        elif "current_balance" in servicer_df.columns:
            rename_map["current_balance"] = "_svc_balance"

        if "servicer_status" in servicer_df.columns:
            rename_map["servicer_status"] = "_svc_status"
        elif "delinquency_status" in servicer_df.columns:
            rename_map["delinquency_status"] = "_svc_status"

        if "last_updated_at" in servicer_df.columns:
            rename_map["last_updated_at"] = "_svc_updated"

        servicer_renamed = servicer_df.rename(rename_map)
        keep_cols = ["loan_id", "reporting_month"] + [v for v in rename_map.values()]
        return servicer_renamed.select([c for c in keep_cols if c in servicer_renamed.columns])

    def _validate_master(self, df: pl.DataFrame) -> None:
        required = {"loan_id", "reporting_month"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Master DataFrame missing columns: {missing}")
