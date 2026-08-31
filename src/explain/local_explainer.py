"""
Local Explainer — local_explainer.py

Computes Local SHAP waterfall breakdowns and contrastive attribution
for individual loan records.
"""

import numpy as np
import pandas as pd
import polars as pl
import shap
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LocalLoanExplainer:
    """
    Computes local SHAP attributions and Contrastive Explanations for individual loans.
    """

    def __init__(self, explainers: Dict[str, Any], features: List[str]):
        """
        Args:
            explainers: Dict mapping target names (e.g. 'default', 'prepay') to trained shap.TreeExplainer objects.
            features: List of feature names in the exact order expected by the explainer.
        """
        self.explainers = explainers
        self.features = features

    def explain_loan(self, loan_record: pl.DataFrame, target: str) -> Dict[str, Any]:
        """
        Computes local SHAP attribution values for a single loan.
        Returns the top 5 positive and negative drivers.
        """
        if target not in self.explainers:
            raise ValueError(f"Explainer for {target} not found.")

        explainer = self.explainers[target]
        pdf = loan_record.select(self.features).to_pandas()
        
        # Calculate SHAP values for the single record
        shap_vals = explainer.shap_values(pdf, check_additivity=False)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]
            
        sv = shap_vals[0]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
            
        # Map values to features
        attributions = list(zip(self.features, sv, pdf.iloc[0].values))
        
        # Sort by impact
        attributions_sorted = sorted(attributions, key=lambda x: x[1], reverse=True)
        
        top_positive = [x for x in attributions_sorted if x[1] > 0][:5]
        top_negative = [x for x in reversed(attributions_sorted) if x[1] < 0][:5]
        
        return {
            "base_value": float(base_value),
            "prediction": float(base_value + np.sum(sv)),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "all_attributions": dict(zip(self.features, sv))
        }

    def explain_contrastive(
        self, 
        loan_record: pl.DataFrame, 
        target_a: str = "default", 
        target_b: str = "prepay"
    ) -> List[Tuple[str, float]]:
        """
        Contrastive Attribution Engine.
        Answers: "Why target_a instead of target_b?"
        Calculates: Delta Phi_i = Phi_i(target_a) - Phi_i(target_b)
        
        Returns:
            Top 3 features responsible for shifting borrower trajectory toward target_a.
        """
        if target_a not in self.explainers or target_b not in self.explainers:
            raise ValueError("Explainers for both targets must be provided.")
            
        pdf = loan_record.select(self.features).to_pandas()
        
        sv_a = self.explainers[target_a].shap_values(pdf, check_additivity=False)
        sv_b = self.explainers[target_b].shap_values(pdf, check_additivity=False)
        
        if isinstance(sv_a, list): sv_a = sv_a[1]
        elif isinstance(sv_a, np.ndarray) and len(sv_a.shape) == 3: sv_a = sv_a[:, :, 1]
            
        if isinstance(sv_b, list): sv_b = sv_b[1]
        elif isinstance(sv_b, np.ndarray) and len(sv_b.shape) == 3: sv_b = sv_b[:, :, 1]
            
        sv_a = sv_a[0]
        sv_b = sv_b[0]
        
        delta_phi = sv_a - sv_b
        
        attributions = list(zip(self.features, delta_phi))
        attributions_sorted = sorted(attributions, key=lambda x: x[1], reverse=True)
        
        # Return top 3 drivers pushing towards target_a over target_b
        return attributions_sorted[:3]

    def export_local_waterfall(self, explanation: Dict[str, Any], output_path: str):
        """
        Renders an interactive Plotly waterfall chart for the local explanation.
        """
        # We manually build a waterfall for Plotly
        base = explanation["base_value"]
        all_attrs = explanation["all_attributions"]
        
        # Sort by absolute impact to show biggest movers last
        sorted_attrs = sorted(all_attrs.items(), key=lambda x: abs(x[1]))
        
        # To avoid clutter, combine small impacts into 'Rest'
        top_n = 10
        if len(sorted_attrs) > top_n:
            small_attrs = sorted_attrs[:-top_n]
            big_attrs = sorted_attrs[-top_n:]
            rest_val = sum(x[1] for x in small_attrs)
            plot_attrs = [("Rest of features", rest_val)] + big_attrs
        else:
            plot_attrs = sorted_attrs
            
        labels = ["Base Value"] + [x[0] for x in plot_attrs] + ["Prediction"]
        values = [base] + [x[1] for x in plot_attrs] + [explanation["prediction"]]
        
        # For plotly waterfall:
        measure = ["absolute"] + ["relative"] * len(plot_attrs) + ["total"]
        
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=measure,
            x=labels,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Local SHAP Waterfall Explanation",
            showlegend=False,
            template="plotly_white"
        )
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        logger.info(f"Local waterfall chart saved to {output_path}")
