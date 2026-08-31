"""
Competing Risk Engine — competing_risks.py

Implements Cumulative Incidence Functions (CIF) and Cause-Specific Cox Hazard
models for the Default vs. Prepayment competing risk framework.

Financial Context:
    In an asset-backed security pool, two competing events can terminate a loan:
        - Default    (Event 1): Signals credit loss; drives CDR and loss severity.
        - Prepayment (Event 2): Drives CPR, shortens cash flow duration, affects
                                WAM and IO strip valuations.

    Standard Kaplan-Meier / single-event Cox models treat the competing event
    as ordinary censoring, OVERESTIMATING the subdistribution hazard of each event.
    Fine-Gray / CIF models correctly account for the mutual exclusivity of events.
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lifelines import AalenJohansenFitter, CoxPHFitter
from src.utils.logger import get_logger

logger = get_logger(__name__)

EVENT_DEFAULT = 1
EVENT_PREPAY = 2


class CompetingRiskEngine:
    """
    Fits Aalen-Johansen CIF estimators and cause-specific Cox PH models
    for Default vs. Prepayment competing risks.

    Outputs multi-horizon cumulative risk vectors that are injected into the
    Feature Store as downstream meta-features for the GBDT ensemble.
    """

    def __init__(self, penalizer: float = 0.01):
        """
        Args:
            penalizer: L2 regularization for cause-specific Cox models.
        """
        self.penalizer = penalizer
        self.cif_default: Optional[AalenJohansenFitter] = None
        self.cif_prepay: Optional[AalenJohansenFitter] = None
        self.cox_default: Optional[CoxPHFitter] = None
        self.cox_prepay: Optional[CoxPHFitter] = None
        logger.info("CompetingRiskEngine initialized.")

    # ------------------------------------------------------------------
    # CIF Estimation (Aalen-Johansen)
    # ------------------------------------------------------------------

    def fit_cif(self, survival_df: pl.DataFrame) -> "CompetingRiskEngine":
        """
        Fits Aalen-Johansen Cumulative Incidence Functions for both events.

        The AJ estimator correctly handles the competing risk structure —
        P(Default by t) + P(Prepay by t) + P(Survive past t) = 1.0 at all t.

        Args:
            survival_df: Loan-level survival frame with columns:
                         duration, event_status (0=Censored, 1=Default, 2=Prepay).

        Returns:
            Self (for chaining).
        """
        pdf = survival_df.to_pandas()
        logger.info(f"Fitting Aalen-Johansen CIF on {len(pdf):,} loans...")

        self.cif_default = AalenJohansenFitter(calculate_variance=True)
        self.cif_default.fit(
            durations=pdf["duration"],
            event_observed=pdf["event_status"],
            event_of_interest=EVENT_DEFAULT,
        )

        self.cif_prepay = AalenJohansenFitter(calculate_variance=True)
        self.cif_prepay.fit(
            durations=pdf["duration"],
            event_observed=pdf["event_status"],
            event_of_interest=EVENT_PREPAY,
        )

        logger.info("CIF estimation complete for Default (Event 1) and Prepayment (Event 2).")
        return self

    # ------------------------------------------------------------------
    # Cause-Specific Cox Models
    # ------------------------------------------------------------------

    def fit_cause_specific_cox(
        self,
        survival_df: pl.DataFrame,
        covariate_cols: List[str],
    ) -> "CompetingRiskEngine":
        """
        Fits two cause-specific Cox PH models.

        Model A — Default as event, Prepayment treated as censored (event=0).
        Model B — Prepayment as event, Default treated as censored (event=0).

        This asymmetric censoring allows each model to learn the covariate
        effects relevant to its specific terminal event, without contamination.

        Args:
            survival_df: Loan-level survival frame.
            covariate_cols: List of numeric covariate columns.

        Returns:
            Self (for chaining).
        """
        pdf = survival_df.to_pandas()
        available_covs = [c for c in covariate_cols if c in pdf.columns]

        if not available_covs:
            raise ValueError("No valid covariate columns for cause-specific Cox.")

        base_cols = ["duration"] + available_covs

        # Model A: Default Cox — prepayment censored
        logger.info("Fitting Cause-Specific Cox A (Default event, Prepayment censored)...")
        df_a = pdf[base_cols].copy()
        df_a["event"] = (pdf["event_status"] == EVENT_DEFAULT).astype(int)
        df_a = df_a.dropna()

        self.cox_default = CoxPHFitter(penalizer=self.penalizer)
        self.cox_default.fit(df_a, duration_col="duration", event_col="event")
        logger.info(f"  Cox-A C-index: {self.cox_default.concordance_index_:.4f}")

        # Model B: Prepayment Cox — default censored
        logger.info("Fitting Cause-Specific Cox B (Prepayment event, Default censored)...")
        df_b = pdf[base_cols].copy()
        df_b["event"] = (pdf["event_status"] == EVENT_PREPAY).astype(int)
        df_b = df_b.dropna()

        self.cox_prepay = CoxPHFitter(penalizer=self.penalizer)
        self.cox_prepay.fit(df_b, duration_col="duration", event_col="event")
        logger.info(f"  Cox-B C-index: {self.cox_prepay.concordance_index_:.4f}")

        return self

    # ------------------------------------------------------------------
    # Multi-Horizon Prediction
    # ------------------------------------------------------------------

    def predict_cumulative_risk(
        self,
        loan_features: pd.DataFrame,
        time_horizons: List[int] = [12, 24, 36, 60],
    ) -> pd.DataFrame:
        """
        Predicts cumulative Default and Prepayment risk at multiple time horizons.

        Uses cause-specific Cox survival functions S_A(t) and S_B(t) to
        approximate CIF as: CIF_k(t) ≈ 1 - S_k(t) (marginal approximation).
        This is a standard actuarial shorthand for cause-specific predictions.

        Args:
            loan_features: DataFrame with covariate columns matching fitted models.
            time_horizons: List of loan-age month checkpoints to evaluate.
                           Default: [12, 24, 36, 60] months.

        Returns:
            pd.DataFrame: One row per loan, columns:
                          cum_default_risk_Xm, cum_prepay_risk_Xm for each horizon.

        Raises:
            RuntimeError: If cause-specific Cox models are not fitted.
        """
        if self.cox_default is None or self.cox_prepay is None:
            raise RuntimeError("Cause-specific Cox models not fitted. Call fit_cause_specific_cox() first.")

        results: Dict[str, np.ndarray] = {}

        for t in time_horizons:
            # Survival probability S(t) for each loan
            s_default = self.cox_default.predict_survival_function(
                loan_features, times=[t]
            ).values[0]  # shape: (n_loans,)

            s_prepay = self.cox_prepay.predict_survival_function(
                loan_features, times=[t]
            ).values[0]

            # CIF ≈ 1 - S(t)
            results[f"cum_default_risk_{t}m"] = np.clip(1.0 - s_default, 0.0, 1.0)
            results[f"cum_prepay_risk_{t}m"] = np.clip(1.0 - s_prepay, 0.0, 1.0)

        return pd.DataFrame(results, index=loan_features.index)

    def export_to_feature_store(
        self,
        loan_features: pd.DataFrame,
        feature_store_path: str = "data/processed/feature_matrix.parquet",
        time_horizons: List[int] = [12, 24, 36, 60],
    ) -> str:
        """
        Appends competing risk meta-features to the existing feature store parquet.

        Meta-features added: cum_default_risk_12m, cum_prepay_risk_12m,
        cum_default_risk_36m, cum_prepay_risk_36m, etc.

        Args:
            loan_features: DataFrame of loan covariates (index = loan_id).
            feature_store_path: Path to existing feature store parquet.
            time_horizons: Time horizons to compute.

        Returns:
            str: Path to the updated feature store.
        """
        risk_df = self.predict_cumulative_risk(loan_features, time_horizons)
        risk_pl = pl.from_pandas(risk_df.reset_index())

        store_path = Path(feature_store_path)
        if store_path.exists():
            existing = pl.read_parquet(str(store_path))
            # Join on loan_id if available
            if "loan_id" in existing.columns and "loan_id" in risk_pl.columns:
                updated = existing.join(risk_pl, on="loan_id", how="left")
            else:
                # Horizontal concat as a fallback
                updated = pl.concat([existing, risk_pl], how="horizontal")
        else:
            updated = risk_pl

        updated.write_parquet(str(store_path), compression="snappy")
        logger.info(f"Feature store updated with competing risk meta-features → {store_path}")
        return str(store_path)

    # ------------------------------------------------------------------
    # Model Diagnostics
    # ------------------------------------------------------------------

    def concordance_indices(self) -> Dict[str, float]:
        """
        Returns concordance indices for both cause-specific Cox models.

        Returns:
            Dict with 'cox_default_cindex' and 'cox_prepay_cindex'.
        """
        result = {}
        if self.cox_default:
            result["cox_default_cindex"] = float(self.cox_default.concordance_index_)
        if self.cox_prepay:
            result["cox_prepay_cindex"] = float(self.cox_prepay.concordance_index_)
        return result
