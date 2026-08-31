"""
Baseline Survival Models — baseline_models.py

Kaplan-Meier non-parametric survival estimators and semi-parametric
Cox Proportional Hazards models for structured finance loan portfolios.

Financial Context:
    Kaplan-Meier curves reveal the "survival" (non-termination) probability
    of loan cohorts over time. Cox PH models quantify how loan attributes
    (credit score, LTV, vintage) multiplicatively shift the baseline hazard.
    Hazard ratios > 1.0 indicate elevated risk; < 1.0 indicate protection.
"""

import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Tuple
from lifelines import KaplanMeierFitter, CoxPHFitter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineSurvivalEngine:
    """
    Fits Kaplan-Meier and Cox Proportional Hazards models to loan survival cohorts.

    Treats event_status in {1, 2} as a single terminal event for baseline
    estimation. Stratified KM curves reveal vintage and credit cohort divergence.
    """

    def __init__(self, penalizer: float = 0.01):
        """
        Args:
            penalizer: L2 regularization strength for Cox model.
                       Prevents numerical instability on correlated covariates.
                       Default 0.01 follows actuarial best practice.
        """
        self.penalizer = penalizer
        self.km_overall: Optional[KaplanMeierFitter] = None
        self.km_stratified: Dict[str, Dict[str, KaplanMeierFitter]] = {}
        self.cox_model: Optional[CoxPHFitter] = None
        logger.info(f"BaselineSurvivalEngine initialized (penalizer={penalizer})")

    def fit_kaplan_meier(
        self,
        survival_df: pl.DataFrame,
        stratify_by: Optional[List[str]] = None,
    ) -> "BaselineSurvivalEngine":
        """
        Fits overall and optionally stratified Kaplan-Meier curves.

        For KM estimation, prepayment (event_status=2) is treated equivalently
        to default as a terminal event. This produces an overall loan termination
        curve — a useful baseline for understanding portfolio aging behavior.

        Args:
            survival_df: Loan-level survival frame (from SurvivalCohortTransformer).
                         Must contain: duration, event_status.
            stratify_by: List of column names to stratify KM curves by.
                         Typical values: ['origination_vintage', 'credit_score_band']

        Returns:
            Self (for method chaining).
        """
        pdf = survival_df.to_pandas()
        # Any non-censored event is treated as terminal for overall KM
        pdf["_event_observed"] = (pdf["event_status"] > 0).astype(int)

        logger.info("Fitting overall Kaplan-Meier curve...")
        self.km_overall = KaplanMeierFitter(label="Overall Portfolio")
        self.km_overall.fit(
            durations=pdf["duration"],
            event_observed=pdf["_event_observed"],
        )

        median_survival = self.km_overall.median_survival_time_
        logger.info(f"KM fit complete. Median survival time: {median_survival:.1f} months")

        # Stratified KM curves
        if stratify_by:
            for col in stratify_by:
                if col not in pdf.columns:
                    logger.warning(f"Stratification column '{col}' not found. Skipping.")
                    continue

                self.km_stratified[col] = {}
                groups = pdf[col].dropna().unique()
                logger.info(f"Fitting stratified KM for '{col}' across {len(groups)} groups...")

                for group_val in groups:
                    mask = pdf[col] == group_val
                    subset = pdf[mask]
                    if len(subset) < 30:
                        logger.warning(f"  Skipping '{group_val}' — only {len(subset)} records.")
                        continue
                    kmf = KaplanMeierFitter(label=str(group_val))
                    kmf.fit(
                        durations=subset["duration"],
                        event_observed=subset["_event_observed"],
                    )
                    self.km_stratified[col][str(group_val)] = kmf

        return self

    def fit_cox(
        self,
        survival_df: pl.DataFrame,
        covariate_cols: List[str],
        event_col: int = 1,
    ) -> "BaselineSurvivalEngine":
        """
        Fits a regularized Cox Proportional Hazards model.

        Schoenfeld residual test is run post-fit to audit proportional
        hazards assumption. Violations are logged with actionable warnings
        rather than silently accepted.

        Args:
            survival_df: Loan-level survival frame.
            covariate_cols: List of numeric/encoded feature columns.
            event_col: Which event_status value to treat as the event of interest.
                       Default 1 = Default. Prepayment (2) is treated as censored.

        Returns:
            Self (for method chaining).
        """
        pdf = survival_df.to_pandas()
        pdf["_event"] = (pdf["event_status"] == event_col).astype(int)

        # Select only valid covariates
        available_covs = [c for c in covariate_cols if c in pdf.columns]
        if not available_covs:
            raise ValueError("No valid covariate columns found in survival DataFrame.")

        cox_data = pdf[["duration", "_event"] + available_covs].dropna()
        logger.info(
            f"Fitting Cox PH model on {len(cox_data):,} records "
            f"with {len(available_covs)} covariates (event={event_col})..."
        )

        self.cox_model = CoxPHFitter(penalizer=self.penalizer)
        self.cox_model.fit(
            cox_data,
            duration_col="duration",
            event_col="_event",
        )

        # Audit proportional hazards assumption
        logger.info("Checking proportional hazards assumption via Schoenfeld residuals...")
        try:
            ph_results = self.cox_model.check_assumptions(
                cox_data, p_value_threshold=0.05, show_plots=False
            )
        except Exception as e:
            logger.warning(f"PH assumption check failed (non-fatal): {e}")

        # Log hazard ratios
        hr_summary = self.cox_model.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%"]]
        logger.info(f"Cox Hazard Ratios:\n{hr_summary.to_string()}")

        return self

    def get_hazard_ratios(self) -> pd.DataFrame:
        """
        Returns the hazard ratio summary table from the fitted Cox model.

        Returns:
            pd.DataFrame: Columns — exp(coef), 95% CI lower/upper, p-value.

        Raises:
            RuntimeError: If Cox model has not been fitted yet.
        """
        if self.cox_model is None:
            raise RuntimeError("Cox model not yet fitted. Call fit_cox() first.")
        return self.cox_model.summary[
            ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
        ].sort_values("exp(coef)", ascending=False)

    def concordance_index(self) -> float:
        """
        Returns the concordance index (C-index) of the fitted Cox model.

        C-index of 0.5 = random. 1.0 = perfect. >0.65 is acceptable in
        credit risk applications with heavy censoring.

        Returns:
            float: Harrell's C-index.
        """
        if self.cox_model is None:
            raise RuntimeError("Cox model not yet fitted.")
        return float(self.cox_model.concordance_index_)
