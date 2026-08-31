"""
Anomaly Score Fusion Engine — score_fusion.py

Combines three tiers of anomaly evidence into a single calibrated 0–100
composite score and derives all submission-template exception columns.

Tier 1 — Deterministic Rules   : Hard accounting violations (binary, definitive)
Tier 2 — Servicer Reconciliation: Multi-source data conflicts (binary)
Tier 3 — ML Ensemble           : Unsupervised Isolation Forest + Autoencoder

Fusion Formula:
    anomaly_score = clip(
        50.0 * rule_violation_flag +
        30.0 * servicer_conflict_flag +
        20.0 * ml_percentile_score,
        0.0, 100.0
    )

This weighted scheme reflects operational priorities:
    - Rule violations are the most severe — they represent provably incorrect data.
    - Servicer conflicts require resolution before any distribution can be signed.
    - ML scores capture soft multivariate patterns not addressable by hard rules.
"""

import numpy as np
import polars as pl
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Fusion weights (must sum to 100 for interpretable score scale)
WEIGHT_RULE: float = 50.0
WEIGHT_RECONCILIATION: float = 30.0
WEIGHT_ML: float = 20.0

# Action and exception thresholds
THRESHOLD_REJECT: float = 85.0
THRESHOLD_TRIAGE: float = 60.0
THRESHOLD_SEVERE_ML_PERCENTILE: float = 0.98  # 98th percentile ML score


class AnomalyFusionEngine:
    """
    Derives all anomaly-related submission columns from three input tiers.

    Produces: anomaly_score, exception_required, exception_type, recommended_action
    """

    def __init__(self):
        logger.info("AnomalyFusionEngine initialized.")

    def fuse(
        self,
        df: pl.DataFrame,
        rule_violation_col: str = "rule_violation_count",
        servicer_conflict_col: str = "servicer_discrepancy_flag",
        iso_score_col: Optional[str] = "iso_anomaly_score",
        ae_score_col: Optional[str] = "ae_anomaly_score",
    ) -> pl.DataFrame:
        """
        Computes the composite anomaly score and derives all exception columns.

        Args:
            df: DataFrame containing rule violation counts, servicer flags,
                and ML anomaly scores.
            rule_violation_col: Column with rule violation count (int).
            servicer_conflict_col: Column with servicer discrepancy flag (bool).
            iso_score_col: Column with Isolation Forest score [0, 1].
            ae_score_col: Column with Autoencoder reconstruction score [0, 1].

        Returns:
            pl.DataFrame: Augmented with:
                anomaly_score, exception_required, exception_type, recommended_action
        """
        logger.info(f"Fusing anomaly scores for {df.height:,} records...")

        # ── Tier 1: Rule violation flag ───────────────────────────────────
        if rule_violation_col in df.columns:
            df = df.with_columns(
                (pl.col(rule_violation_col) > 0).cast(pl.Float64).alias("_rule_flag")
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias("_rule_flag"))

        # ── Tier 2: Servicer conflict flag ────────────────────────────────
        if servicer_conflict_col in df.columns:
            df = df.with_columns(
                pl.col(servicer_conflict_col).cast(pl.Float64).alias("_svc_flag")
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias("_svc_flag"))

        # ── Tier 3: ML composite score [0, 1] ────────────────────────────
        # Average Isolation Forest and Autoencoder if both available
        ml_parts = []
        for col in [iso_score_col, ae_score_col]:
            if col and col in df.columns:
                ml_parts.append(pl.col(col).cast(pl.Float64))

        if ml_parts:
            ml_avg = sum(ml_parts) / len(ml_parts)
            df = df.with_columns(ml_avg.alias("_ml_score"))
        else:
            df = df.with_columns(pl.lit(0.0).alias("_ml_score"))

        # ── Composite Score ───────────────────────────────────────────────
        df = df.with_columns(
            (
                WEIGHT_RULE * pl.col("_rule_flag") +
                WEIGHT_RECONCILIATION * pl.col("_svc_flag") +
                WEIGHT_ML * pl.col("_ml_score")
            ).clip(0.0, 100.0)
            .alias("anomaly_score")
        )

        # ── exception_required ────────────────────────────────────────────
        df = df.with_columns(
            (
                (pl.col("anomaly_score") >= THRESHOLD_TRIAGE) |
                (pl.col("_rule_flag") == 1.0)
            ).alias("exception_required")
        )

        # ── exception_type (hierarchical) ─────────────────────────────────
        df = df.with_columns(
            pl.when(pl.col("_rule_flag") == 1.0)
            .then(pl.lit("Data Logic Error"))
            .when(pl.col("_svc_flag") == 1.0)
            .then(pl.lit("Servicer Discrepancy"))
            .when(
                pl.col("_ml_score") >= THRESHOLD_SEVERE_ML_PERCENTILE
            )
            .then(pl.lit("Severe Deterioration"))
            .when(pl.col("exception_required"))
            .then(pl.lit("Severe Deterioration"))  # catch-all for high ML-driven exceptions
            .otherwise(pl.lit("None"))
            .alias("exception_type")
        )

        # ── recommended_action ────────────────────────────────────────────
        df = df.with_columns(
            pl.when(pl.col("anomaly_score") >= THRESHOLD_REJECT)
            .then(pl.lit("Reject/Repurchase"))
            .when(pl.col("anomaly_score") >= THRESHOLD_TRIAGE)
            .then(pl.lit("Manual Triage"))
            .otherwise(pl.lit("Auto-Approve"))
            .alias("recommended_action")
        )

        # ── confidence: 1 - normalized uncertainty across tiers ──────────
        # Confidence is higher when multiple tiers agree (all high or all low)
        df = df.with_columns(
            (1.0 - (pl.col("anomaly_score") / 100.0 * (1.0 - pl.col("anomaly_score") / 100.0) * 4.0))
            .clip(0.0, 1.0)
            .alias("confidence")
        )

        # Drop internal staging columns
        df = df.drop(["_rule_flag", "_svc_flag", "_ml_score"])

        approved = df.filter(pl.col("exception_required").not_()).height
        triaged = df.filter(pl.col("recommended_action") == "Manual Triage").height
        rejected = df.filter(pl.col("recommended_action") == "Reject/Repurchase").height
        logger.info(
            f"Fusion complete: "
            f"Auto-Approve={approved:,}, Manual Triage={triaged:,}, Reject={rejected:,}"
        )
        return df
