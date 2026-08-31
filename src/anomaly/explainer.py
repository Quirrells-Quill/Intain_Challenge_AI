"""
Anomaly Explainer — explainer.py

Generates human-interpretable, semicolon-delimited top driver strings
for each exception record to populate the submission template's `top_drivers` column.

Two attribution strategies:
    1. Rule-based records: Return the activated rule code strings directly
       (deterministic, no ML required).
    2. ML outlier records: Use TreeSHAP on the Isolation Forest to extract
       the top 3 features most responsible for driving the anomaly score.

Output format: "HighCurrentLTV;DTIJump;RULE_BALANCE_OVERFLOW"
"""

import numpy as np
import polars as pl
import shap
from typing import List, Optional
from src.anomaly.ml_detectors import UnsupervisedAnomalyStack
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum background samples for SHAP TreeExplainer (performance guard)
SHAP_BACKGROUND_SAMPLES = 200


class AnomalyExplainer:
    """
    Attributes anomaly scores to top contributing features using SHAP and rule codes.

    For the submission template, top_drivers is always a semicolon-delimited
    string of exactly up to 3 driver codes per record.
    """

    def __init__(
        self,
        anomaly_stack: UnsupervisedAnomalyStack,
        feature_names: List[str],
    ):
        """
        Args:
            anomaly_stack: Fitted UnsupervisedAnomalyStack (for SHAP access).
            feature_names: Column names corresponding to the feature matrix used.
        """
        self.anomaly_stack = anomaly_stack
        self.feature_names = feature_names
        self._shap_explainer: Optional[shap.TreeExplainer] = None
        logger.info(f"AnomalyExplainer initialized with {len(feature_names)} features.")

    # ------------------------------------------------------------------
    # SHAP Explainer Setup
    # ------------------------------------------------------------------

    def build_shap_explainer(self, X_background: np.ndarray) -> "AnomalyExplainer":
        """
        Initializes TreeSHAP for the fitted Isolation Forest.

        Uses a random subsample of background records to approximate the
        marginal feature distribution — critical for performance on large pools.

        Args:
            X_background: Array of background records (already scaled).

        Returns:
            Self (for chaining).
        """
        if self.anomaly_stack.iso_forest is None:
            raise RuntimeError("Isolation Forest not fitted. Cannot build SHAP explainer.")

        n = min(SHAP_BACKGROUND_SAMPLES, X_background.shape[0])
        idx = np.random.choice(X_background.shape[0], n, replace=False)
        background = X_background[idx]

        logger.info(f"Building TreeSHAP explainer with {n} background samples...")
        self._shap_explainer = shap.TreeExplainer(
            self.anomaly_stack.iso_forest,
            data=background,
            feature_perturbation="interventional",
        )
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_batch(
        self,
        df: pl.DataFrame,
        X_scaled: np.ndarray,
        rule_driver_col: str = "rule_drivers",
        exception_type_col: str = "exception_type",
        top_k: int = 3,
    ) -> pl.Series:
        """
        Generates top_drivers strings for all records in the DataFrame.

        Strategy:
            - Records with 'Data Logic Error': use rule codes from rule_driver_col.
            - All other exception records: use TreeSHAP feature attributions.
            - Non-exception records: return 'None'.

        Args:
            df: DataFrame with exception_type and optional rule_drivers columns.
            X_scaled: Scaled feature matrix (row-aligned with df).
            rule_driver_col: Column containing semicolon-delimited rule codes.
            exception_type_col: Column with exception type classification.
            top_k: Number of top driver features to return.

        Returns:
            pl.Series: top_drivers strings, one per record.
        """
        exception_types = df[exception_type_col].to_list() if exception_type_col in df.columns else ["None"] * df.height
        rule_drivers = df[rule_driver_col].to_list() if rule_driver_col in df.columns else ["None"] * df.height

        # Pre-compute SHAP values for ML-flagged records if explainer is ready
        shap_values: Optional[np.ndarray] = None
        if self._shap_explainer is not None:
            try:
                logger.info("Computing TreeSHAP values for anomaly attribution...")
                shap_values = self._shap_explainer.shap_values(X_scaled)
                if isinstance(shap_values, list):
                    # IsolationForest SHAP returns list[array]; take absolute mean
                    shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            except Exception as e:
                logger.warning(f"SHAP computation failed (non-fatal): {e}")

        top_drivers_list: List[str] = []
        for i, (ex_type, rule_str) in enumerate(zip(exception_types, rule_drivers)):

            if ex_type == "Data Logic Error" and rule_str and rule_str != "None":
                # Rule-based: return rule codes directly
                parts = rule_str.split(";")[:top_k]
                top_drivers_list.append(";".join(parts))

            elif ex_type in ("Servicer Discrepancy", "Severe Deterioration", "None") and shap_values is not None:
                # ML-based: extract top k features by absolute SHAP contribution
                row_shap = np.abs(shap_values[i])
                top_indices = np.argsort(row_shap)[::-1][:top_k]
                driver_codes = [self._format_feature_code(self.feature_names[j]) for j in top_indices]
                top_drivers_list.append(";".join(driver_codes))

            elif rule_str and rule_str != "None":
                top_drivers_list.append(";".join(rule_str.split(";")[:top_k]))

            else:
                top_drivers_list.append("None")

        return pl.Series("top_drivers", top_drivers_list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_feature_code(feature_name: str) -> str:
        """
        Converts a snake_case feature name into a CamelCase driver code
        matching the submission template convention.

        Example: 'current_ltv' -> 'CurrentLtv'
        """
        return "".join(word.capitalize() for word in feature_name.split("_"))
