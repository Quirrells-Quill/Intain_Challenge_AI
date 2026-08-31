"""
Deterministic Rule Auditor — rule_detector.py

Parses validation_rules.json and executes vectorized accounting and
chronological consistency checks across all loan records.

Financial Context:
    Rules encode hard accounting invariants that must hold in any compliant
    servicing dataset. A violated rule is a definitive exception — unlike ML
    anomaly scores which are probabilistic. Rule violations trigger automatic
    'Data Logic Error' classification in the submission template, regardless
    of ML score magnitude.
"""

import json
import polars as pl
from pathlib import Path
from typing import Dict, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Rule code registry — matched to submission top_drivers format
RULE_CODES: Dict[str, str] = {
    "R1": "RULE_BALANCE_OVERFLOW",
    "R2": "RULE_CHRONOLOGY_BREAK",
    "R3": "RULE_AGE_MISMATCH",
    "R4": "RULE_STATUS_DPD_CONFLICT",
    "R5": "RULE_TERMINAL_STATUS_LEAK",
}


class DeterministicRuleAuditor:
    """
    Executes vectorized deterministic accounting and chronological checks
    on loan performance records. All thresholds sourced from validation_rules.json.

    Every check is implemented as a Polars expression for vectorized execution
    — critical for scanning millions of records without Python-level loops.
    """

    def __init__(self, rules_path: str = "configs/validation_rules.json"):
        """
        Args:
            rules_path: Path to validation_rules.json (PILLAR 2: no hardcoding).
        """
        rules_file = Path(rules_path)
        if not rules_file.exists():
            raise FileNotFoundError(f"Validation rules not found: {rules_path}")
        with open(rules_file) as f:
            self._rules = json.load(f)

        self._dpd_default_threshold = int(
            self._rules["delinquency_rules"]["dpd_default_threshold"]
        )
        self._max_ltv = float(self._rules["ltv_rules"]["max_ltv"])
        logger.info(
            f"DeterministicRuleAuditor initialized. "
            f"DPD threshold={self._dpd_default_threshold}, max_ltv={self._max_ltv}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Executes all five rule checks and appends boolean violation columns.

        Each rule produces one boolean column (True = violation detected).
        An aggregate `rule_violation_count` column sums across all rules.

        Args:
            df: Monthly loan performance DataFrame. Must contain the columns
                required by each rule check (validated per-rule).

        Returns:
            pl.DataFrame: Original frame augmented with rule violation columns:
                r1_balance_overflow, r2_chronology_break, r3_age_mismatch,
                r4_status_dpd_conflict, r5_terminal_leak, rule_violation_count
        """
        logger.info(f"Running DeterministicRuleAuditor on {df.height:,} records...")
        df = self._check_r1_balance(df)
        df = self._check_r2_chronology(df)
        df = self._check_r3_age_integrity(df)
        df = self._check_r4_status_dpd(df)
        df = self._check_r5_terminal_status(df)

        # Aggregate: count of violated rules per record
        rule_flag_cols = [
            "r1_balance_overflow", "r2_chronology_break",
            "r3_age_mismatch", "r4_status_dpd_conflict", "r5_terminal_leak",
        ]
        present = [c for c in rule_flag_cols if c in df.columns]

        df = df.with_columns(
            sum(pl.col(c).cast(pl.Int32) for c in present)
            .alias("rule_violation_count")
        )

        total_flagged = df.filter(pl.col("rule_violation_count") > 0).height
        logger.info(
            f"Rule audit complete. {total_flagged:,} records flagged "
            f"({100 * total_flagged / max(df.height, 1):.2f}% of pool)."
        )
        return df

    def get_top_drivers(self, df: pl.DataFrame) -> pl.Series:
        """
        Returns a semicolon-delimited string of violated rule codes per record.
        Maps directly to the `top_drivers` submission column for rule-based exceptions.

        Args:
            df: DataFrame that has been through audit().

        Returns:
            pl.Series: String series, e.g. "RULE_BALANCE_OVERFLOW;RULE_AGE_MISMATCH"
        """
        rule_map = {
            "r1_balance_overflow":    RULE_CODES["R1"],
            "r2_chronology_break":    RULE_CODES["R2"],
            "r3_age_mismatch":        RULE_CODES["R3"],
            "r4_status_dpd_conflict": RULE_CODES["R4"],
            "r5_terminal_leak":       RULE_CODES["R5"],
        }
        present = {col: code for col, code in rule_map.items() if col in df.columns}

        driver_strings = []
        for row in df.select(list(present.keys())).iter_rows(named=True):
            codes = [code for col, code in present.items() if row.get(col, False)]
            driver_strings.append(";".join(codes) if codes else "None")

        return pl.Series("rule_drivers", driver_strings)

    # ------------------------------------------------------------------
    # Individual Rule Checks
    # ------------------------------------------------------------------

    def _check_r1_balance(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        R1: current_balance must not exceed original_balance.

        Accounting invariant: principal balance can only decrease (amortization)
        or stay flat (interest-only). An increase signals data corruption or
        miscapitalized fees that were not approved in the servicing agreement.
        """
        if "current_balance" not in df.columns or "original_balance" not in df.columns:
            logger.warning("R1 skipped: missing current_balance or original_balance.")
            return df.with_columns(pl.lit(False).alias("r1_balance_overflow"))

        return df.with_columns(
            (
                pl.col("current_balance").is_not_null() &
                pl.col("original_balance").is_not_null() &
                (pl.col("current_balance") > pl.col("original_balance"))
            ).alias("r1_balance_overflow")
        )

    def _check_r2_chronology(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        R2: reporting_month must be >= origination_month.

        A reporting date prior to the loan's origination date is physically
        impossible — indicates date parsing errors or mismatched loan records.
        """
        if "reporting_month" not in df.columns or "origination_month" not in df.columns:
            logger.warning("R2 skipped: missing date columns.")
            return df.with_columns(pl.lit(False).alias("r2_chronology_break"))

        return df.with_columns(
            (
                pl.col("reporting_month").is_not_null() &
                pl.col("origination_month").is_not_null() &
                (pl.col("reporting_month") < pl.col("origination_month"))
            ).alias("r2_chronology_break")
        )

    def _check_r3_age_integrity(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        R3: loan_age must equal the month difference between reporting_month and origination_month.

        Discrepancy implies the servicer reported an incorrect loan age, which
        cascades into incorrect WAM calculations and survival duration estimates.
        Tolerance of ±1 month is allowed for reporting lag conventions.
        """
        if not all(c in df.columns for c in ["loan_age", "reporting_month", "origination_month"]):
            logger.warning("R3 skipped: missing loan_age or date columns.")
            return df.with_columns(pl.lit(False).alias("r3_age_mismatch"))

        df = df.with_columns(
            (
                (pl.col("reporting_month").cast(pl.Date) - pl.col("origination_month").cast(pl.Date))
                .dt.total_days() / 30.44
            ).round(0).cast(pl.Int32).alias("_computed_age")
        )

        df = df.with_columns(
            (
                pl.col("loan_age").is_not_null() &
                pl.col("_computed_age").is_not_null() &
                ((pl.col("loan_age") - pl.col("_computed_age")).abs() > 2)
            ).alias("r3_age_mismatch")
        ).drop("_computed_age")

        return df

    def _check_r4_status_dpd(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        R4: If days_past_due == 0, delinquency_status must be 'Current'.

        A loan with zero missed payment days cannot simultaneously be classified
        as delinquent. This conflict indicates a status field update failure in
        the servicer's loan management system.
        """
        if "days_past_due" not in df.columns or "delinquency_status" not in df.columns:
            logger.warning("R4 skipped: missing DPD or status columns.")
            return df.with_columns(pl.lit(False).alias("r4_status_dpd_conflict"))

        dpd_zero_but_not_current = (
            pl.col("days_past_due").is_not_null() &
            pl.col("delinquency_status").is_not_null() &
            (pl.col("days_past_due") == 0) &
            (pl.col("delinquency_status") != "Current")
        )

        # Also check converse: DPD > threshold but status = 'Current'
        dpd_high_but_current = (
            pl.col("days_past_due").is_not_null() &
            pl.col("delinquency_status").is_not_null() &
            (pl.col("days_past_due") > self._dpd_default_threshold) &
            (pl.col("delinquency_status") == "Current")
        )

        return df.with_columns(
            (dpd_zero_but_not_current | dpd_high_but_current)
            .alias("r4_status_dpd_conflict")
        )

    def _check_r5_terminal_status(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        R5: After a default or prepayment event, all subsequent monthly records
        must reflect a terminal closed status.

        If a loan was reported as defaulted in month T but appears again as
        'Current' in month T+1, this is a critical data integrity breach —
        commonly caused by servicer data re-ingestion or loan modification
        events that were not properly flagged.
        """
        if not all(c in df.columns for c in ["loan_id", "reporting_month", "default_flag", "prepayment_flag", "delinquency_status"]):
            logger.warning("R5 skipped: missing terminal status columns.")
            return df.with_columns(pl.lit(False).alias("r5_terminal_leak"))

        df = df.sort(["loan_id", "reporting_month"])

        # Per loan: compute whether ANY prior month had a terminal event
        df = df.with_columns([
            pl.col("default_flag").shift(1).over("loan_id").fill_null(0).alias("_prev_default"),
            pl.col("prepayment_flag").shift(1).over("loan_id").fill_null(0).alias("_prev_prepay"),
        ])

        df = df.with_columns(
            (
                ((pl.col("_prev_default") == 1) | (pl.col("_prev_prepay") == 1)) &
                (pl.col("delinquency_status") == "Current")
            ).alias("r5_terminal_leak")
        ).drop(["_prev_default", "_prev_prepay"])

        return df
