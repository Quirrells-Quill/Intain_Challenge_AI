"""
Fairness Auditor — fairness_auditor.py

Evaluates model predictions across protected or sensitive proxy subgroups
to audit for disparate impact, False Positive Rates, and calibration stability.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SubgroupFairnessAuditor:
    """
    Audits the portfolio for demographic, geographic, and vintage disparities.
    """

    def __init__(self, target_col: str = "default_flag", prob_col: str = "prob_default", threshold: float = 0.50):
        self.target_col = target_col
        self.prob_col = prob_col
        self.threshold = threshold
        self.subgroups = ["origination_vintage", "state", "credit_score_band"]

    def audit(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        Computes fairness metrics across subgroups.
        
        Metrics:
            - FPR (False Positive Rate)
            - DIR (Disparate Impact Ratio) based on positive prediction rates.
        """
        logger.info("Executing Subgroup Bias & Fairness Audit...")
        
        # Add hard prediction based on threshold
        df = df.with_columns(
            (pl.col(self.prob_col) >= self.threshold).cast(pl.Int32).alias("predicted_class")
        )
        
        portfolio_fpr = self._compute_fpr(df)
        portfolio_ppr = df["predicted_class"].mean() # Positive Prediction Rate
        
        ledger = {}
        
        for group in self.subgroups:
            if group not in df.columns:
                continue
                
            results = []
            grouped = df.group_by(group)
            
            for name, sub_df in grouped:
                # We extract the group value from the tuple returned by polars group_by
                group_val = name[0] if isinstance(name, tuple) else name
                
                fpr = self._compute_fpr(sub_df)
                ppr = sub_df["predicted_class"].mean()
                
                dir_score = ppr / max(portfolio_ppr, 1e-6)
                
                # Flag if FPR is > 1.5x the portfolio baseline
                flagged = fpr > (1.5 * portfolio_fpr) if portfolio_fpr > 0 else False
                
                results.append({
                    "subgroup": group,
                    "value": str(group_val),
                    "loan_count": sub_df.height,
                    "fpr": float(fpr),
                    "dir": float(dir_score),
                    "flagged_disparity": flagged
                })
                
            ledger[group] = pl.DataFrame(results).sort("fpr", descending=True)
            
        return ledger

    def _compute_fpr(self, df: pl.DataFrame) -> float:
        """Computes False Positive Rate: FP / (FP + TN)"""
        # Negatives are target == 0
        negatives = df.filter(pl.col(self.target_col) == 0)
        if negatives.height == 0:
            return 0.0
        
        fps = negatives.filter(pl.col("predicted_class") == 1).height
        return fps / negatives.height
