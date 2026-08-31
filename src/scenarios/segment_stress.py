"""
Segment Stress Analyzer — segment_stress.py

Stratified stress impact analyzer computing relative vulnerability across
critical portfolio slices (Vintage, Credit Score Band, State, Servicer).
"""

import polars as pl
from typing import Dict, List, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SegmentStressAnalyzer:
    """
    Computes risk delta between scenarios (e.g., Base vs Adverse) segmented by 
    portfolio dimensions to identify concentration vulnerabilities.
    """

    def __init__(self):
        self.segments = [
            "origination_vintage",
            "credit_score_band",
            "state",
            "servicer_name"
        ]

    def analyze_vulnerability(
        self,
        base_df: pl.DataFrame,
        base_scores: pl.DataFrame,
        adverse_scores: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculates the relative default risk increase for every distinct segment bucket.

        Args:
            base_df: Loan-level features containing segment columns.
            base_scores: DataFrame with 'prob_default' from the Base scenario.
            adverse_scores: DataFrame with 'prob_default' from the Adverse scenario.

        Returns:
            pl.DataFrame: Segment-level vulnerabilities ranked by relative risk delta.
        """
        logger.info("Analyzing segment vulnerability (Adverse vs Base).")

        # Combine features with predictions
        df = base_df.with_columns([
            base_scores["prob_default"].alias("base_prob"),
            adverse_scores["prob_default"].alias("adv_prob"),
        ])

        results = []

        for seg_col in self.segments:
            if seg_col not in df.columns:
                logger.warning(f"Segment column missing: {seg_col}")
                continue
            
            # Aggregate by segment bucket
            agg_df = df.group_by(seg_col).agg([
                pl.len().alias("loan_count"),
                pl.col("base_prob").mean().alias("base_default_rate"),
                pl.col("adv_prob").mean().alias("adverse_default_rate")
            ])
            
            # Compute relative delta: (Adv - Base) / Base
            # Using pl.max to avoid division by zero
            agg_df = agg_df.with_columns(
                ((pl.col("adverse_default_rate") - pl.col("base_default_rate")) / 
                 pl.col("base_default_rate").clip(lower_bound=1e-6)
                ).alias("relative_risk_delta")
            )
            
            # Collect and append dimension info
            for row in agg_df.iter_rows(named=True):
                if row["loan_count"] < 10:  # Skip micro-segments to avoid noise
                    continue
                results.append({
                    "dimension": seg_col,
                    "segment_value": str(row[seg_col]),
                    "loan_count": row["loan_count"],
                    "base_default_rate": row["base_default_rate"],
                    "adverse_default_rate": row["adverse_default_rate"],
                    "relative_risk_delta": row["relative_risk_delta"]
                })

        if not results:
            return pl.DataFrame()

        vulnerability_df = pl.DataFrame(results).sort("relative_risk_delta", descending=True)
        return vulnerability_df

    def get_top_vulnerable_segments(self, vulnerability_df: pl.DataFrame, top_k: int = 3) -> List[Dict]:
        """Returns the top K most vulnerable segments."""
        if vulnerability_df.height == 0:
            return []
        
        top_k_df = vulnerability_df.head(top_k)
        return top_k_df.to_dicts()
