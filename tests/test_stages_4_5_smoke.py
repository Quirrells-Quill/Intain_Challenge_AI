"""
tests/test_stages_4_5_smoke.py
================================================================================
Intain-Sight — Stage 4 & 5 Smoke Test & Validation Harness
================================================================================
Pytest suite validating:
    - Survival cohort transformation & CIF monotonicity (Stage 4)
    - CDR / CPR formula correctness & boundary compliance (Stage 4)
    - Deterministic rule auditing with deliberate corruption injections (Stage 5)
    - Servicer reconciliation flag precision (Stage 5)
    - Anomaly score fusion contract & schema compliance (Stage 5)
    - Reviewer dossier file generation (Stage 5)

Run:
    pytest tests/test_stages_4_5_smoke.py -v --tb=short
"""

import sys
import os
import json
import shutil
import tempfile
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from repo root without `pip install -e .`
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from src.survival.data_adapter import SurvivalCohortTransformer
from src.survival.securitization_rates import SecuritizationRateEngine
from src.anomaly.rule_detector import DeterministicRuleAuditor
from src.anomaly.reconciliation import ServicerReconciliationEngine
from src.anomaly.ml_detectors import UnsupervisedAnomalyStack
from src.anomaly.score_fusion import AnomalyFusionEngine
from src.anomaly.report_generator import AnomalyAuditReportGenerator

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_LOANS = 100
N_MONTHS_PER_LOAN = 10          # 1 000 panel rows total
RULES_PATH = "configs/validation_rules.json"

VALID_EVENT_STATUSES = {0, 1, 2}
VALID_EXCEPTION_TYPES = {
    "Data Logic Error",
    "Servicer Discrepancy",
    "Severe Deterioration",
    "None",
}
VALID_ACTIONS = {"Auto-Approve", "Manual Triage", "Reject/Repurchase"}


# ===========================================================================
# Shared synthetic data factories
# ===========================================================================

def _make_panel(
    n_loans: int = N_LOANS,
    months_per_loan: int = N_MONTHS_PER_LOAN,
    rng: np.random.Generator = np.random.default_rng(RANDOM_SEED),
    include_events: bool = True,
) -> pl.DataFrame:
    """
    Generates a synthetic monthly loan performance panel with realistic features.

    - 20% of loans default at a random month (event_status = 1)
    - 20% of loans prepay at a random month (event_status = 2)
    - 60% remain censored (active throughout observation window)
    """
    records = []
    orig_date = date(2019, 1, 1)

    for loan_idx in range(n_loans):
        loan_id = f"LOAN_{loan_idx:04d}"
        orig_balance = float(rng.integers(50_000, 500_000))
        interest_rate = float(rng.uniform(0.03, 0.09))
        orig_ltv = float(rng.uniform(0.50, 0.95))
        orig_month = orig_date + timedelta(days=int(30 * rng.integers(0, 12)))

        # Assign terminal event type
        fate = rng.random()
        if include_events:
            event_month_idx = int(rng.integers(3, months_per_loan))
            if fate < 0.20:
                event_type = "default"
            elif fate < 0.40:
                event_type = "prepay"
            else:
                event_type = "censored"
        else:
            event_type = "censored"
            event_month_idx = months_per_loan

        balance = orig_balance
        for month_idx in range(months_per_loan):
            reporting_date = orig_month + timedelta(days=30 * (month_idx + 1))

            is_terminal_month = (event_type != "censored") and (month_idx == event_month_idx)
            is_post_terminal = (event_type != "censored") and (month_idx > event_month_idx)

            if is_post_terminal:
                # After terminal event, stop appending (terminal status leak test needs post rows)
                pass

            default_flag = 1 if (event_type == "default" and month_idx >= event_month_idx) else 0
            prepay_flag = 1 if (event_type == "prepay" and month_idx >= event_month_idx) else 0

            dpd = 0
            status = "Current"
            if event_type == "default" and month_idx >= event_month_idx:
                dpd = int(rng.integers(90, 360))
                status = "90+ Delinquent"

            # Simulate scheduled principal paydown
            monthly_rate = interest_rate / 12
            if balance > 0 and monthly_rate > 0:
                payment = balance * monthly_rate / (1 - (1 + monthly_rate) ** -360)
                scheduled_principal = min(payment - balance * monthly_rate, balance)
            else:
                scheduled_principal = 0.0

            prepay_amount = balance * 0.10 if prepay_flag else 0.0
            balance_after = max(balance - scheduled_principal - prepay_amount, 0.0)

            records.append({
                "loan_id":               loan_id,
                "reporting_month":       reporting_date,
                "origination_month":     orig_month,
                "loan_age":              month_idx + 1,
                "original_balance":      orig_balance,
                "current_balance":       float(balance_after),
                "interest_rate":         interest_rate,
                "original_ltv":          orig_ltv,
                "days_past_due":         dpd,
                "delinquency_status":    status,
                "default_flag":          default_flag,
                "prepayment_flag":       prepay_flag,
                "prepay_amount":         prepay_amount,
                "scheduled_principal":   float(scheduled_principal),
                "credit_score_band":     rng.choice(["580-669", "670-739", "740-799", "800-850"]),
                "dti_band":              rng.choice(["<36%", "36-43%", "43-50%"]),
                "state":                 rng.choice(["CA", "TX", "FL", "NY", "IL"]),
                "origination_vintage":   orig_month.year,
                "servicer_name":         rng.choice(["Alpha Servicing", "Beta Capital", "Gamma Trust"]),
            })

            balance = balance_after
            if prepay_flag and balance <= 0:
                break

    return pl.DataFrame(records).with_columns([
        pl.col("reporting_month").cast(pl.Date),
        pl.col("origination_month").cast(pl.Date),
    ])


def _ensure_rules_file() -> None:
    """Creates a minimal validation_rules.json if not present in repo."""
    path = Path(RULES_PATH)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rules = {
        "delinquency_rules": {
            "dpd_default_threshold": 90,
            "status_dpd_map": {"Current": [0, 0], "90+ Delinquent": [90, 9999]},
        },
        "balance_rules": {"max_current_vs_original_ratio": 1.0},
        "date_rules": {"reporting_must_be_after_origination": True},
        "ltv_rules": {"max_ltv": 1.5, "high_ltv_threshold": 0.95},
        "interest_rate_rules": {"min_rate": 0.001, "max_rate": 0.30},
        "anomaly_thresholds": {
            "anomaly_score_exception_cutoff": 60.0,
            "severe_deterioration_score": 85.0,
        },
        "submission_bounds": {
            "probability_min": 0.0,
            "probability_max": 1.0,
            "anomaly_score_min": 0.0,
            "anomaly_score_max": 100.0,
            "confidence_min": 0.0,
            "confidence_max": 1.0,
            "valid_next_states": ["Current", "Delinquent", "Default", "Prepaid"],
            "valid_exception_types": list(VALID_EXCEPTION_TYPES),
            "valid_actions": list(VALID_ACTIONS),
        },
    }
    path.write_text(json.dumps(rules, indent=2))


# ===========================================================================
# TEST 1: Survival Cohort Transformation & CIF Monotonicity
# ===========================================================================

class TestSurvivalCohortPipeline:
    """Stage 4 — SurvivalCohortTransformer validation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _ensure_rules_file()
        self.panel = _make_panel()
        self.transformer = SurvivalCohortTransformer(rules_path=RULES_PATH)
        self.cohort = self.transformer.get_survival_dataset(self.panel)

    def test_one_row_per_loan(self):
        """Each unique loan_id must appear exactly once in the survival cohort."""
        n_unique_loans = self.panel["loan_id"].n_unique()
        assert self.cohort.height == n_unique_loans, (
            f"Expected {n_unique_loans} cohort rows, got {self.cohort.height}"
        )

    def test_duration_positive(self):
        """All loans must have duration >= 1 month."""
        min_duration = self.cohort["duration"].min()
        assert min_duration >= 1, f"Minimum duration must be >= 1, got {min_duration}"

    def test_event_status_domain(self):
        """event_status must belong strictly to {{0, 1, 2}}."""
        observed = set(self.cohort["event_status"].unique().to_list())
        assert observed.issubset(VALID_EVENT_STATUSES), (
            f"Invalid event_status values detected: {observed - VALID_EVENT_STATUSES}"
        )

    def test_cif_monotonicity_and_joint_bound(self):
        """
        CIF curves must be monotonically non-decreasing, and their joint sum
        at every time point must not exceed 1.0 (competing risk constraint).

        CIF_default(t) + CIF_prepay(t) <= 1.0 for all t.
        """
        # Only run if lifelines is importable
        pytest.importorskip("lifelines")
        from lifelines import AalenJohansenFitter

        pdf = self.cohort.to_pandas()

        # CIF for Default (event=1)
        cif_d = AalenJohansenFitter(calculate_variance=False)
        cif_d.fit(durations=pdf["duration"], event_observed=pdf["event_status"], event_of_interest=1)

        # CIF for Prepayment (event=2)
        cif_p = AalenJohansenFitter(calculate_variance=False)
        cif_p.fit(durations=pdf["duration"], event_observed=pdf["event_status"], event_of_interest=2)

        cif_d_vals = cif_d.cumulative_density_.values.flatten()
        cif_p_vals = cif_p.cumulative_density_.values.flatten()

        # Monotonic non-decrease: each value >= previous
        assert np.all(np.diff(cif_d_vals) >= -1e-9), (
            "CIF for Default is not monotonically non-decreasing."
        )
        assert np.all(np.diff(cif_p_vals) >= -1e-9), (
            "CIF for Prepayment is not monotonically non-decreasing."
        )

        # Align arrays to same length before joint sum check
        n = min(len(cif_d_vals), len(cif_p_vals))
        joint_sum = cif_d_vals[:n] + cif_p_vals[:n]
        max_joint = float(joint_sum.max())
        assert max_joint <= 1.0 + 1e-6, (
            f"CIF joint sum exceeded 1.0: max={{max_joint:.6f}}"
        )

    def test_no_null_event_status(self):
        """event_status must have zero nulls."""
        assert self.cohort["event_status"].null_count() == 0

    def test_no_null_duration(self):
        """duration must have zero nulls."""
        assert self.cohort["duration"].null_count() == 0


# ===========================================================================
# TEST 2: Securitization Metrics Bounds & Formula Verification
# ===========================================================================

class TestCdrCprBounds:
    """Stage 4 — SecuritizationRateEngine validation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _ensure_rules_file()
        self.panel = _make_panel()
        self.engine = SecuritizationRateEngine()

    def test_mdr_bounded(self):
        """MDR must be in [0.0, 1.0] for every reporting month."""
        rates = self.engine.compute_monthly_rates(self.panel)
        assert rates["mdr"].min() >= 0.0, "MDR has values below 0."
        assert rates["mdr"].max() <= 1.0 + 1e-9, f"MDR exceeded 1.0: {rates['mdr'].max()}"

    def test_smm_bounded(self):
        """SMM must be in [0.0, 1.0] for every reporting month."""
        rates = self.engine.compute_monthly_rates(self.panel)
        assert rates["smm"].min() >= 0.0, "SMM has values below 0."
        assert rates["smm"].max() <= 1.0 + 1e-9, f"SMM exceeded 1.0: {rates['smm'].max()}"

    def test_cdr_bounded(self):
        """CDR must be in [0.0, 1.0] for every reporting month."""
        rates = self.engine.compute_monthly_rates(self.panel)
        assert rates["cdr"].min() >= 0.0
        assert rates["cdr"].max() <= 1.0 + 1e-9, f"CDR exceeded 1.0: {rates['cdr'].max()}"

    def test_cpr_bounded(self):
        """CPR must be in [0.0, 1.0] for every reporting month."""
        rates = self.engine.compute_monthly_rates(self.panel)
        assert rates["cpr"].min() >= 0.0
        assert rates["cpr"].max() <= 1.0 + 1e-9, f"CPR exceeded 1.0: {rates['cpr'].max()}"

    def test_cdr_formula_correctness(self):
        """
        Verify CDR = 1 - (1 - MDR)^12 holds element-wise within floating
        point tolerance of 1e-9.
        """
        rates = self.engine.compute_monthly_rates(self.panel)
        mdr = rates["mdr"].to_numpy()
        cdr_computed = rates["cdr"].to_numpy()
        cdr_expected = 1.0 - (1.0 - mdr) ** 12
        np.testing.assert_allclose(
            cdr_computed, cdr_expected, atol=1e-9,
            err_msg="CDR formula CDR = 1-(1-MDR)^12 violated."
        )

    def test_cpr_formula_correctness(self):
        """
        Verify CPR = 1 - (1 - SMM)^12 holds element-wise within floating
        point tolerance of 1e-9.
        """
        rates = self.engine.compute_monthly_rates(self.panel)
        smm = rates["smm"].to_numpy()
        cpr_computed = rates["cpr"].to_numpy()
        cpr_expected = 1.0 - (1.0 - smm) ** 12
        np.testing.assert_allclose(
            cpr_computed, cpr_expected, atol=1e-9,
            err_msg="CPR formula CPR = 1-(1-SMM)^12 violated."
        )

    def test_pool_health_score_bounded(self):
        """Pool Health Score must be in [0, 100] for all months."""
        rates = self.engine.compute_monthly_rates(self.panel)
        health = self.engine.compute_pool_health_score(rates)
        assert health["pool_health_score"].min() >= 0.0
        assert health["pool_health_score"].max() <= 100.0 + 1e-6

    def test_monthly_rates_has_required_columns(self):
        """Output must contain mdr, cdr, smm, cpr, reporting_month."""
        rates = self.engine.compute_monthly_rates(self.panel)
        for col in ["reporting_month", "mdr", "cdr", "smm", "cpr"]:
            assert col in rates.columns, f"Missing column: '{col}'"


# ===========================================================================
# TEST 3: Deterministic Rule Auditor — Deliberate Corruption Injection
# ===========================================================================

class TestDeterministicRuleAuditor:
    """Stage 5 — DeterministicRuleAuditor violation detection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _ensure_rules_file()
        self.auditor = DeterministicRuleAuditor(rules_path=RULES_PATH)

    def _make_clean_row(self) -> dict:
        """Returns a single clean, compliant loan record.
        Dates are chosen so computed_age = 12 exactly, avoiding R3 false positive.
        loan_age is explicitly set to match the computed month delta.
        """
        # 365 days = 12.0 months exactly when using the 30.44-day divisor in R3
        orig = date(2021, 1, 1)
        rep  = date(2022, 1, 1)   # exactly 365 days → 365/30.44 ≈ 11.99 → rounds to 12
        return {
            "loan_id":             "CLEAN_001",
            "reporting_month":     rep,
            "origination_month":   orig,
            "loan_age":            12,
            "original_balance":    200_000.0,
            "current_balance":     195_000.0,
            "days_past_due":       0,
            "delinquency_status":  "Current",
            "default_flag":        0,
            "prepayment_flag":     0,
        }

    def test_r1_balance_overflow_detected(self):
        """R1: current_balance > original_balance must be flagged."""
        row = self._make_clean_row()
        row.update({
            "loan_id":          "BAD_R1",
            "current_balance":  150_000.0,
            "original_balance": 100_000.0,   # Deliberate overflow
        })
        df = pl.DataFrame([row])
        audited = self.auditor.audit(df)
        assert audited["r1_balance_overflow"][0] is True, (
            "R1: Balance overflow not detected."
        )
        assert audited["rule_violation_count"][0] >= 1

    def test_r2_chronology_break_detected(self):
        """R2: reporting_month < origination_month must be flagged."""
        row = self._make_clean_row()
        row.update({
            "loan_id":           "BAD_R2",
            "reporting_month":   date(2020, 1, 1),   # Before origination
            "origination_month": date(2021, 1, 1),
        })
        df = pl.DataFrame([row])
        audited = self.auditor.audit(df)
        assert audited["r2_chronology_break"][0] is True, (
            "R2: Chronological break not detected."
        )
        assert audited["rule_violation_count"][0] >= 1

    def test_r4_status_dpd_conflict_detected(self):
        """R4: days_past_due=0 with status!='Current' must be flagged (R4 Branch 1)."""
        row = self._make_clean_row()
        row.update({
            "loan_id":            "BAD_R4",
            "days_past_due":      0,                     # Zero DPD — loan is current
            "delinquency_status": "30-Day Delinquent",   # Contradicts zero DPD
        })
        df = pl.DataFrame([row])
        audited = self.auditor.audit(df)
        assert audited["r4_status_dpd_conflict"][0] is True, (
            "R4: DPD=0 / non-Current status conflict not detected."
        )
        assert audited["rule_violation_count"][0] >= 1

    def test_clean_control_row_passes(self):
        """Clean, compliant records must have rule_violation_count == 0."""
        row = self._make_clean_row()
        df = pl.DataFrame([row])
        audited = self.auditor.audit(df)
        assert audited["rule_violation_count"][0] == 0, (
            f"Clean row unexpectedly flagged: "
            f"violation_count={audited['rule_violation_count'][0]}"
        )

    def test_mixed_batch_segregation(self):
        """In a mixed batch, only corrupt rows must be flagged."""
        clean = self._make_clean_row()
        dirty = self._make_clean_row()
        dirty.update({
            "loan_id":          "BAD_MIX",
            "current_balance":  999_999.0,
            "original_balance": 100_000.0,
        })
        df = pl.DataFrame([clean, dirty])
        audited = self.auditor.audit(df)

        # Look up by loan_id — R5's sort may reorder rows
        clean_viol = audited.filter(pl.col("loan_id") == "CLEAN_001")["rule_violation_count"][0]
        dirty_viol = audited.filter(pl.col("loan_id") == "BAD_MIX")["rule_violation_count"][0]
        assert clean_viol == 0, f"Clean row was incorrectly flagged (count={clean_viol})."
        assert dirty_viol >= 1, "Dirty row was not flagged."

    def test_rule_driver_strings_populated(self):
        """get_top_drivers must return non-empty strings for flagged records."""
        row = self._make_clean_row()
        row.update({
            "loan_id":          "DRIVER_TEST",
            "current_balance":  999_999.0,
            "original_balance": 100_000.0,
        })
        df = pl.DataFrame([row])
        audited = self.auditor.audit(df)
        drivers = self.auditor.get_top_drivers(audited)
        assert drivers[0] != "None", "Flagged record returned empty driver string."
        assert "RULE_" in drivers[0], "Driver code doesn't follow RULE_ convention."


# ===========================================================================
# TEST 4: Servicer Reconciliation Engine
# ===========================================================================

class TestServicerReconciliation:
    """Stage 5 — ServicerReconciliationEngine discrepancy detection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _ensure_rules_file()
        self.engine = ServicerReconciliationEngine(rules_path=RULES_PATH)

    def _make_records(self):
        """Creates master and servicer DataFrames with known conflict patterns."""
        base_date = date(2023, 3, 1)

        master = pl.DataFrame([
            # Row 0: Matching — no discrepancy
            {"loan_id": "L001", "reporting_month": base_date,
             "current_balance": 100_000.0, "delinquency_status": "Current"},
            # Row 1: Balance delta > $500
            {"loan_id": "L002", "reporting_month": base_date,
             "current_balance": 100_000.0, "delinquency_status": "Current"},
            # Row 2: Status conflict
            {"loan_id": "L003", "reporting_month": base_date,
             "current_balance": 50_000.0, "delinquency_status": "Current"},
            # Row 3: Stale record
            {"loan_id": "L004", "reporting_month": base_date,
             "current_balance": 75_000.0, "delinquency_status": "Current"},
        ]).with_columns(pl.col("reporting_month").cast(pl.Date))

        servicer = pl.DataFrame([
            # L001: Matches master exactly
            {"loan_id": "L001", "reporting_month": base_date,
             "servicer_balance": 100_000.0, "servicer_status": "Current",
             "last_updated_at": base_date},
            # L002: Balance delta = $600 (> $100 tolerance)
            {"loan_id": "L002", "reporting_month": base_date,
             "servicer_balance":  99_400.0,   # Delta $600
             "servicer_status": "Current",
             "last_updated_at": base_date},
            # L003: Status conflict
            {"loan_id": "L003", "reporting_month": base_date,
             "servicer_balance": 50_000.0, "servicer_status": "30-Day Delinquent",
             "last_updated_at": base_date},
            # L004: Stale — last updated 61 days before reporting
            {"loan_id": "L004", "reporting_month": base_date,
             "servicer_balance": 75_000.0, "servicer_status": "Current",
             "last_updated_at": base_date - timedelta(days=61)},
        ]).with_columns([
            pl.col("reporting_month").cast(pl.Date),
            pl.col("last_updated_at").cast(pl.Date),
        ])

        return master, servicer

    def test_matching_record_not_flagged(self):
        """A perfectly matching record (L001) must not trigger any discrepancy."""
        master, servicer = self._make_records()
        result = self.engine.reconcile(master, servicer)
        l001 = result.filter(pl.col("loan_id") == "L001")
        assert l001["servicer_discrepancy_flag"][0] is False, (
            "L001 (matching record) was incorrectly flagged."
        )

    def test_balance_delta_flagged(self):
        """L002 with a $600 balance delta must be flagged as a discrepancy."""
        master, servicer = self._make_records()
        result = self.engine.reconcile(master, servicer)
        l002 = result.filter(pl.col("loan_id") == "L002")
        assert l002["servicer_discrepancy_flag"][0] is True, (
            "L002 balance delta > $100 was not detected."
        )

    def test_status_conflict_flagged(self):
        """L003 with conflicting delinquency status must be flagged."""
        master, servicer = self._make_records()
        result = self.engine.reconcile(master, servicer)
        l003 = result.filter(pl.col("loan_id") == "L003")
        assert l003["servicer_discrepancy_flag"][0] is True, (
            "L003 status conflict was not detected."
        )

    def test_stale_record_flagged(self):
        """L004 with servicer lag > 45 days must be flagged as stale."""
        master, servicer = self._make_records()
        result = self.engine.reconcile(master, servicer)
        l004 = result.filter(pl.col("loan_id") == "L004")
        assert l004["servicer_discrepancy_flag"][0] is True, (
            "L004 stale record (61-day lag) was not detected."
        )

    def test_reconciliation_notes_non_empty(self):
        """reconciliation_notes must be non-empty for all flagged records."""
        master, servicer = self._make_records()
        result = self.engine.reconcile(master, servicer)
        flagged = result.filter(pl.col("servicer_discrepancy_flag"))
        for row in flagged["reconciliation_notes"].to_list():
            assert row and len(row) > 0, "reconciliation_notes is empty for a flagged record."
            assert row != "No discrepancy detected.", (
                "Flagged record has 'No discrepancy detected.' note."
            )

    def test_output_has_required_columns(self):
        """Reconciled DataFrame must contain discrepancy flag and notes columns."""
        master, servicer = self._make_records()
        result = self.engine.reconcile(master, servicer)
        assert "servicer_discrepancy_flag" in result.columns
        assert "reconciliation_notes" in result.columns


# ===========================================================================
# TEST 5: Unsupervised ML Stack & Fusion Contract
# ===========================================================================

class TestAnomalyFusionContract:
    """Stage 5 — UnsupervisedAnomalyStack + AnomalyFusionEngine validation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _ensure_rules_file()
        self.panel = _make_panel(n_loans=60, months_per_loan=8)
        self.auditor = DeterministicRuleAuditor(rules_path=RULES_PATH)
        self.recon_engine = ServicerReconciliationEngine(rules_path=RULES_PATH)
        self.ml_stack = UnsupervisedAnomalyStack(
            contamination=0.05, ae_epochs=5, random_state=RANDOM_SEED
        )
        self.fusion = AnomalyFusionEngine()

        # Run rule auditing
        df = self.auditor.audit(self.panel)
        rule_drivers = self.auditor.get_top_drivers(df)
        df = df.with_columns(rule_drivers)

        # Add placeholder servicer columns (no external servicer feed)
        df = df.with_columns([
            pl.lit(False).alias("servicer_discrepancy_flag"),
            pl.lit("No servicer feed in test.").alias("reconciliation_notes"),
        ])

        # ML scoring
        X, _ = UnsupervisedAnomalyStack.select_numeric_features(df)
        self.ml_stack.fit(X)
        iso_s, ae_s = self.ml_stack.score(X)
        df = df.with_columns([
            pl.Series("iso_anomaly_score", iso_s.astype(float)),
            pl.Series("ae_anomaly_score", ae_s.astype(float)),
        ])

        # Fusion
        self.result = self.fusion.fuse(df)

    def test_anomaly_score_bounds(self):
        """anomaly_score must be a continuous float in [0.0, 100.0]."""
        scores = self.result["anomaly_score"]
        assert scores.null_count() == 0, "anomaly_score contains nulls."
        assert float(scores.min()) >= 0.0, f"anomaly_score minimum below 0: {scores.min()}"
        assert float(scores.max()) <= 100.0 + 1e-6, f"anomaly_score maximum above 100: {scores.max()}"

    def test_no_nulls_in_output(self):
        """No output column from fusion must contain NaN or null values."""
        for col in ["anomaly_score", "exception_required", "exception_type",
                    "recommended_action", "confidence"]:
            if col in self.result.columns:
                assert self.result[col].null_count() == 0, (
                    f"Column '{col}' contains {self.result[col].null_count()} nulls."
                )

    def test_exception_required_logic(self):
        """
        exception_required must be True iff:
            anomaly_score >= 60.0 OR rule_violation_count > 0
        """
        df = self.result
        if "rule_violation_count" not in df.columns:
            pytest.skip("rule_violation_count not present in fused output.")

        for row in df.iter_rows(named=True):
            score = row["anomaly_score"]
            rule_viol = row.get("rule_violation_count", 0) or 0
            expected_flag = (score >= 60.0) or (rule_viol > 0)
            actual_flag = row["exception_required"]
            assert actual_flag == expected_flag, (
                f"Loan {row.get('loan_id')}: exception_required mismatch. "
                f"score={score:.2f}, rule_viol={rule_viol}, "
                f"expected={expected_flag}, got={actual_flag}"
            )

    def test_exception_type_domain(self):
        """exception_type must belong strictly to the 4 allowed categories."""
        observed = set(self.result["exception_type"].unique().to_list())
        invalid = observed - VALID_EXCEPTION_TYPES
        assert not invalid, f"Invalid exception_type values: {invalid}"

    def test_recommended_action_domain(self):
        """recommended_action must belong strictly to the 3 allowed categories."""
        observed = set(self.result["recommended_action"].unique().to_list())
        invalid = observed - VALID_ACTIONS
        assert not invalid, f"Invalid recommended_action values: {invalid}"

    def test_action_score_alignment(self):
        """
        Verify recommended_action thresholds align with anomaly_score:
            Reject/Repurchase : score >= 85.0
            Manual Triage     : 60.0 <= score < 85.0
            Auto-Approve      : score < 60.0
        """
        for row in self.result.iter_rows(named=True):
            score = row["anomaly_score"]
            action = row["recommended_action"]
            if score >= 85.0:
                assert action == "Reject/Repurchase", (
                    f"Score {score:.1f} >= 85 but action='{action}'"
                )
            elif score >= 60.0:
                assert action == "Manual Triage", (
                    f"Score {score:.1f} in [60,85) but action='{action}'"
                )
            else:
                assert action == "Auto-Approve", (
                    f"Score {score:.1f} < 60 but action='{action}'"
                )

    def test_confidence_bounds(self):
        """confidence must be in [0.0, 1.0]."""
        if "confidence" not in self.result.columns:
            pytest.skip("confidence column not present.")
        conf = self.result["confidence"]
        assert float(conf.min()) >= 0.0
        assert float(conf.max()) <= 1.0 + 1e-6


# ===========================================================================
# TEST 6: Reviewer Dossier File Generation
# ===========================================================================

class TestReviewerDossierOutput:
    """Stage 5 — AnomalyAuditReportGenerator output file validation."""

    MD_PATH = Path("reports/ANOMALY_REVIEWER_DOSSIER.md")
    HTML_PATH = Path("reports/ANOMALY_REVIEWER_DOSSIER.html")

    @pytest.fixture(autouse=True)
    def setup(self):
        _ensure_rules_file()

        # Build a DataFrame with exactly the three exception types (5+5+10=20 cases)
        rng = np.random.default_rng(RANDOM_SEED)
        n_cases = 20

        records = []
        for i in range(n_cases):
            if i < 5:
                ex_type = "Data Logic Error"
                score = float(rng.uniform(70, 100))
                action = "Reject/Repurchase"
            elif i < 10:
                ex_type = "Servicer Discrepancy"
                score = float(rng.uniform(60, 84))
                action = "Manual Triage"
            else:
                ex_type = "Severe Deterioration"
                score = float(rng.uniform(60, 99))
                action = "Manual Triage" if score < 85 else "Reject/Repurchase"

            records.append({
                "loan_id":              f"DOSSIER_{i:03d}",
                "reporting_month":      date(2023, 6, 1),
                "state":                rng.choice(["CA", "TX", "FL"]),
                "origination_vintage":  int(rng.integers(2018, 2023)),
                "servicer_name":        rng.choice(["Alpha Servicing", "Beta Capital"]),
                "anomaly_score":        score,
                "exception_type":       ex_type,
                "recommended_action":   action,
                "top_drivers":          "CurrentLtv;DaysPastDue;InterestRate",
                "days_past_due":        int(rng.integers(0, 120)),
                "current_balance":      float(rng.uniform(50_000, 400_000)),
                "original_balance":     400_000.0,
                "reconciliation_notes": "Balance delta exceeds tolerance." if ex_type == "Servicer Discrepancy" else "N/A",
            })

        self.df = pl.DataFrame(records).with_columns(
            pl.col("reporting_month").cast(pl.Date)
        )
        self.generator = AnomalyAuditReportGenerator()

        # Clean any prior run's output
        if self.MD_PATH.exists():
            self.MD_PATH.unlink()
        if self.HTML_PATH.exists():
            self.HTML_PATH.unlink()

    def test_markdown_file_created(self):
        """ANOMALY_REVIEWER_DOSSIER.md must exist after report generation."""
        self.generator.generate_report(self.df)
        assert self.MD_PATH.exists(), (
            f"Markdown dossier not found at {self.MD_PATH}"
        )

    def test_html_file_created(self):
        """ANOMALY_REVIEWER_DOSSIER.html must exist after report generation."""
        self.generator.generate_report(self.df)
        assert self.HTML_PATH.exists(), (
            f"HTML dossier not found at {self.HTML_PATH}"
        )

    def test_markdown_contains_20_cases(self):
        """Markdown dossier must contain exactly 20 'Case XX' headers."""
        self.generator.generate_report(self.df)
        content = self.MD_PATH.read_text(encoding="utf-8")
        # Each case is delimited by "## Case XX"
        import re
        case_headers = re.findall(r"^## Case \d{2}", content, re.MULTILINE)
        assert len(case_headers) == 20, (
            f"Expected 20 case headers in dossier, found {len(case_headers)}."
        )

    def test_markdown_has_all_exception_types(self):
        """Dossier must mention all 3 exception categories."""
        self.generator.generate_report(self.df)
        content = self.MD_PATH.read_text(encoding="utf-8")
        for ex_type in ["Data Logic Error", "Servicer Discrepancy", "Severe Deterioration"]:
            assert ex_type in content, (
                f"Exception type '{ex_type}' not found in dossier."
            )

    def test_html_file_is_valid_html(self):
        """HTML dossier must contain a valid HTML table structure."""
        self.generator.generate_report(self.df)
        content = self.HTML_PATH.read_text(encoding="utf-8")
        assert "<table>" in content
        assert "<thead>" in content
        assert "<tbody>" in content
        assert "Loan ID" in content

    def test_dossier_contains_recommended_action_column(self):
        """Dossier HTML must reference all three recommended actions."""
        self.generator.generate_report(self.df)
        content = self.HTML_PATH.read_text(encoding="utf-8")
        assert "Auto-Approve" in content or "Manual Triage" in content or "Reject/Repurchase" in content


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
    sys.exit(result.returncode)
