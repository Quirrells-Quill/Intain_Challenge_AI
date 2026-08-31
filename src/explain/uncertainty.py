"""
Epistemic Uncertainty Engine — uncertainty.py

Uses Deep Ensemble variance across diverse GBDT seeds to compute standard deviations 
per prediction, routing high-uncertainty loans directly to human underwriters.
"""

import numpy as np
import polars as pl
from typing import List, Callable, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleUncertaintyEstimator:
    """
    Computes model uncertainty (epistemic) using a panel of independent model outputs.
    """

    def __init__(self, ensemble_predictors: List[Callable[[pl.DataFrame], pl.DataFrame]], target_col: str = "prob_default"):
        """
        Args:
            ensemble_predictors: List of predict functions representing different models/seeds.
            target_col: The probability column to evaluate.
        """
        self.predictors = ensemble_predictors
        self.target_col = target_col
        self.K = len(self.predictors)
        logger.info(f"EnsembleUncertaintyEstimator initialized with {self.K} predictors.")

    def estimate_uncertainty(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Computes mean prediction, standard deviation, and confidence score.
        Flags records requiring human review if std_dev >= 0.12.
        """
        if self.K == 0:
            raise ValueError("No predictors provided to uncertainty estimator.")
            
        logger.info("Computing epistemic uncertainty across ensemble...")
        
        preds_list = []
        for i, predictor in enumerate(self.predictors):
            out = predictor(df)
            preds_list.append(out[self.target_col].to_numpy())
            
        preds_matrix = np.column_stack(preds_list)
        
        mean_p = np.mean(preds_matrix, axis=1)
        std_p = np.std(preds_matrix, axis=1)
        confidence = 1.0 - (2.0 * std_p)
        
        # High uncertainty triage flag
        requires_review = std_p >= 0.12
        
        return df.with_columns([
            pl.Series("ensemble_mean", mean_p),
            pl.Series("ensemble_std", std_p),
            pl.Series("model_confidence", confidence).clip(lower_bound=0.0, upper_bound=1.0),
            pl.Series("uncertainty_review_required", requires_review)
        ])

    def generate_uncertainty_heatmap(self, df: pl.DataFrame, output_path: str):
        """
        Generates a Plotly heatmap comparing predicted probability vs. model uncertainty.
        """
        import plotly.express as px
        import os
        
        if "ensemble_mean" not in df.columns or "ensemble_std" not in df.columns:
            logger.warning("Uncertainty features missing for heatmap.")
            return
            
        pdf = df.select(["ensemble_mean", "ensemble_std", "credit_score_band"]).to_pandas()
        
        fig = px.density_heatmap(
            pdf, x="ensemble_mean", y="ensemble_std", facet_col="credit_score_band",
            title="Epistemic Uncertainty vs Predicted Probability by Credit Band",
            labels={"ensemble_mean": "Predicted Probability", "ensemble_std": "Ensemble Standard Deviation"},
            color_continuous_scale="Viridis"
        )
        # Add the human-review threshold line
        fig.add_hline(y=0.12, line_dash="dash", line_color="red", annotation_text="Human Review Threshold")
        fig.update_layout(template="plotly_white")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        logger.info(f"Uncertainty heatmap saved to {output_path}")
