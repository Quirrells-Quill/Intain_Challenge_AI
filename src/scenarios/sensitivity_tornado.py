"""
Sensitivity Tornado Engine — sensitivity_tornado.py

Generates one-at-a-time (OAT) marginal sensitivity profiles to construct
a feature sensitivity tornado chart.
"""

import os
import polars as pl
import plotly.graph_objects as go
from typing import Callable, Dict, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TornadoSensitivityEngine:
    """
    Measures marginal portfolio sensitivity to individual macro and borrower
    feature shifts to produce a Tornado Chart.
    """

    def __init__(self):
        # Definition of independent shocks
        self.shocks = {
            "Interest Rate (100 bps)": {"col": "interest_rate", "up": 0.0100, "down": -0.0100},
            "Property Value/HPI (10%)": {"col": "current_ltv", "up": -0.10, "down": 0.10}, # +10% HPI = -10% LTV roughly
            "Unemployment/DTI (15%)": {"col": "dti", "up": 15.0, "down": -15.0},
            "Credit Score (30 pts)": {"col": "credit_score", "up": -30, "down": 30}, # For risk: Up shock = lower score (higher risk)
        }

    def compute_sensitivities(
        self,
        df: pl.DataFrame,
        predict_fn: Callable[[pl.DataFrame], pl.DataFrame]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Computes the absolute change in portfolio default rate when individual
        variables are shifted Up or Down.

        Args:
            df: Baseline feature DataFrame.
            predict_fn: Scoring function returning 'prob_default'.

        Returns:
            Dict mapping shock name to (down_impact, up_impact)
        """
        logger.info("Computing marginal sensitivity (Tornado)...")
        
        # 1. Base default rate
        base_scores = predict_fn(df)
        base_rate = base_scores["prob_default"].mean()
        
        sensitivities = {}

        for shock_name, config in self.shocks.items():
            col = config["col"]
            if col not in df.columns:
                logger.warning(f"Column '{col}' missing. Skipping shock '{shock_name}'.")
                continue

            # Up Shock
            df_up = df.with_columns(
                (pl.col(col) + config["up"]).alias(col)
            )
            up_rate = predict_fn(df_up)["prob_default"].mean()

            # Down Shock
            df_down = df.with_columns(
                (pl.col(col) + config["down"]).alias(col)
            )
            down_rate = predict_fn(df_down)["prob_default"].mean()

            # We store the absolute delta from base
            sensitivities[shock_name] = (down_rate - base_rate, up_rate - base_rate)

        return sensitivities

    def generate_tornado_chart(
        self, 
        sensitivities: Dict[str, Tuple[float, float]], 
        output_path: str = "reports/figures/scenarios/tornado_sensitivity.html"
    ):
        """
        Renders a Plotly Diverging Bar Chart and saves to HTML.
        """
        if not sensitivities:
            logger.warning("No sensitivities to plot.")
            return

        # Sort by maximum absolute impact
        sorted_shocks = sorted(
            sensitivities.items(),
            key=lambda item: max(abs(item[1][0]), abs(item[1][1]))
        )

        y_labels = [k for k, v in sorted_shocks]
        down_vals = [v[0] * 100 for k, v in sorted_shocks] # Convert to percentage points
        up_vals = [v[1] * 100 for k, v in sorted_shocks]

        fig = go.Figure()

        # Add Down (Favorable/Unfavorable) bars
        fig.add_trace(go.Bar(
            y=y_labels,
            x=down_vals,
            name='Down Shock',
            orientation='h',
            marker_color='mediumseagreen'
        ))

        # Add Up bars
        fig.add_trace(go.Bar(
            y=y_labels,
            x=up_vals,
            name='Up Shock',
            orientation='h',
            marker_color='indianred'
        ))

        fig.update_layout(
            title="Portfolio Default Sensitivity (Tornado Chart)",
            xaxis_title="Change in Portfolio Default Rate (Percentage Points)",
            barmode='relative',
            yaxis_autorange='reversed',
            template="plotly_white"
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        logger.info(f"Tornado chart saved to {output_path}")
