"""
Survival Visualizer — risk_heatmaps.py

Generates interactive Plotly visualizations for structured finance survival analysis:
    - Competing CIF curves (Default vs. Prepayment) with confidence bands
    - Dual-axis CDR / CPR time series
    - 2D hazard heatmap: credit band × loan age → default probability surface
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Dict
from lifelines import AalenJohansenFitter
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("reports/figures/survival")


class SurvivalVisualizer:
    """
    Exports self-contained interactive HTML charts for survival analysis outputs.
    All figures are written to reports/figures/survival/.
    """

    def __init__(self, output_dir: str = str(OUTPUT_DIR)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SurvivalVisualizer initialized. Output: {self.output_dir}")

    # ------------------------------------------------------------------
    # Chart 1: Competing CIF Curves
    # ------------------------------------------------------------------

    def plot_competing_cif(
        self,
        cif_default: AalenJohansenFitter,
        cif_prepay: AalenJohansenFitter,
        filename: str = "competing_cif.html",
    ) -> str:
        """
        Renders competing Cumulative Incidence Functions for Default and Prepayment
        with shaded 95% confidence intervals.

        The non-overlap of confidence bands provides visual evidence that the
        two competing risks are statistically distinguishable at each loan age.

        Args:
            cif_default: Fitted AalenJohansenFitter for Default (event=1).
            cif_prepay: Fitted AalenJohansenFitter for Prepayment (event=2).
            filename: Output HTML filename.

        Returns:
            str: Absolute path to the exported HTML file.
        """
        fig = go.Figure()

        # Default CIF
        t_d = cif_default.cumulative_density_.index.values
        ci_d = cif_default.cumulative_density_.values.flatten()
        ci_d_lo = cif_default.confidence_interval_cumulative_density_.iloc[:, 0].values
        ci_d_hi = cif_default.confidence_interval_cumulative_density_.iloc[:, 1].values

        fig.add_trace(go.Scatter(
            x=np.concatenate([t_d, t_d[::-1]]),
            y=np.concatenate([ci_d_hi, ci_d_lo[::-1]]),
            fill="toself", fillcolor="rgba(220,50,50,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Default 95% CI", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=t_d, y=ci_d,
            mode="lines", line=dict(color="crimson", width=2.5),
            name="CIF — Default (Event 1)",
        ))

        # Prepayment CIF
        t_p = cif_prepay.cumulative_density_.index.values
        ci_p = cif_prepay.cumulative_density_.values.flatten()
        ci_p_lo = cif_prepay.confidence_interval_cumulative_density_.iloc[:, 0].values
        ci_p_hi = cif_prepay.confidence_interval_cumulative_density_.iloc[:, 1].values

        fig.add_trace(go.Scatter(
            x=np.concatenate([t_p, t_p[::-1]]),
            y=np.concatenate([ci_p_hi, ci_p_lo[::-1]]),
            fill="toself", fillcolor="rgba(30,120,200,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Prepayment 95% CI", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=t_p, y=ci_p,
            mode="lines", line=dict(color="royalblue", width=2.5),
            name="CIF — Prepayment (Event 2)",
        ))

        fig.update_layout(
            title="Competing Risk: Cumulative Incidence Functions (Default vs. Prepayment)",
            xaxis_title="Loan Age (Months)",
            yaxis_title="Cumulative Incidence Probability",
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            legend=dict(x=0.02, y=0.98),
            template="plotly_white",
            height=550,
        )

        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        logger.info(f"Competing CIF chart saved → {output_path}")
        return str(output_path)

    # ------------------------------------------------------------------
    # Chart 2: CDR / CPR Dual-Axis Time Series
    # ------------------------------------------------------------------

    def plot_cdr_cpr_timeseries(
        self,
        monthly_rates: pl.DataFrame,
        filename: str = "cdr_cpr_timeseries.html",
    ) -> str:
        """
        Dual-axis Plotly chart overlaying CDR (left axis) and CPR (right axis)
        on the same time axis.

        The divergence or convergence of CDR and CPR trajectories is a leading
        indicator of pool health regime shifts (e.g., rising defaults + falling
        prepayments signals trapped distressed borrowers).

        Args:
            monthly_rates: Output of SecuritizationRateEngine.compute_monthly_rates().
            filename: Output HTML filename.

        Returns:
            str: Path to exported HTML.
        """
        pdf = monthly_rates.sort("reporting_month").to_pandas()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=pdf["reporting_month"], y=pdf["cdr"],
                name="CDR (Conditional Default Rate)",
                line=dict(color="crimson", width=2),
                mode="lines+markers", marker=dict(size=4),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=pdf["reporting_month"], y=pdf["cpr"],
                name="CPR (Conditional Prepayment Rate)",
                line=dict(color="steelblue", width=2, dash="dash"),
                mode="lines+markers", marker=dict(size=4),
            ),
            secondary_y=True,
        )

        fig.update_xaxes(title_text="Reporting Month")
        fig.update_yaxes(title_text="CDR (Annualized)", tickformat=".1%", secondary_y=False)
        fig.update_yaxes(title_text="CPR (Annualized)", tickformat=".1%", secondary_y=True)
        fig.update_layout(
            title="Pool-Level CDR vs. CPR — Historical Time Series",
            template="plotly_white",
            height=500,
            legend=dict(x=0.01, y=0.99),
        )

        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        logger.info(f"CDR/CPR time series chart saved → {output_path}")
        return str(output_path)

    # ------------------------------------------------------------------
    # Chart 3: Hazard Heatmap (Credit Band × Loan Age)
    # ------------------------------------------------------------------

    def plot_hazard_heatmap(
        self,
        survival_df: pl.DataFrame,
        age_bins: int = 12,
        filename: str = "hazard_heatmap.html",
    ) -> str:
        """
        2D heatmap: X-axis = loan_age_months buckets, Y-axis = credit_score_band,
        Z-value = empirical default rate within each cell.

        Reveals interaction effects between borrower credit quality and loan
        seasoning. Diagonal bright spots indicate 'vintage stress clusters'
        common in sub-prime ABS pools.

        Args:
            survival_df: Loan-level survival frame from SurvivalCohortTransformer.
            age_bins: Number of equal-width loan age bins to create.
            filename: Output HTML filename.

        Returns:
            str: Path to exported HTML.
        """
        df = survival_df.with_columns([
            (pl.col("duration") // (pl.col("duration").max() // age_bins + 1))
            .alias("age_bin")
        ])

        pdf = df.to_pandas()
        pdf["age_bin_label"] = (pdf["age_bin"] * (pdf["duration"].max() // age_bins)).astype(int).astype(str) + "m"

        pivot = (
            pdf[pdf["event_status"] == 1]
            .groupby(["credit_score_band", "age_bin_label"])
            .size()
            .unstack(fill_value=0)
        )

        # Normalize by total loans per cell
        total = pdf.groupby(["credit_score_band", "age_bin_label"]).size().unstack(fill_value=1)
        rate_matrix = (pivot / total).fillna(0)

        fig = go.Figure(data=go.Heatmap(
            z=rate_matrix.values,
            x=rate_matrix.columns.tolist(),
            y=rate_matrix.index.tolist(),
            colorscale="YlOrRd",
            colorbar=dict(title="Default Rate", tickformat=".1%"),
            zmin=0,
            zmax=rate_matrix.values.max(),
        ))

        fig.update_layout(
            title="Default Hazard Heatmap: Credit Band × Loan Age",
            xaxis_title="Loan Age (Months)",
            yaxis_title="Credit Score Band",
            template="plotly_white",
            height=500,
        )

        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        logger.info(f"Hazard heatmap saved → {output_path}")
        return str(output_path)
