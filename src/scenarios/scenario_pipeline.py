"""
Scenario & Stress Pipeline — scenario_pipeline.py

Orchestrates the end-to-end Monte Carlo simulation, segment stress testing,
tornado sensitivity analysis, and automated reporting.
"""

import os
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Callable, Optional, Dict

from src.scenarios.scenario_mapper import MacroScenarioMapper
from src.scenarios.monte_carlo import MonteCarloSimulator
from src.scenarios.segment_stress import SegmentStressAnalyzer
from src.scenarios.sensitivity_tornado import TornadoSensitivityEngine
from src.scenarios.scenario_narrator import ScenarioNarrativeBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScenarioPipeline:
    def __init__(self, n_simulations: int = 1000, random_seed: int = 42):
        self.mapper = MacroScenarioMapper()
        self.simulator = MonteCarloSimulator(self.mapper, n_simulations, random_seed)
        self.stress_analyzer = SegmentStressAnalyzer()
        self.tornado_engine = TornadoSensitivityEngine()
        self.narrator = ScenarioNarrativeBuilder()

        # Artifact directories
        self.reports_dir = "reports"
        self.figures_dir = os.path.join(self.reports_dir, "figures", "scenarios")
        self.data_dir = os.path.join("data", "processed")
        
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def _default_proxy_model(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        A proxy deterministic scoring function if no ML model is provided.
        Simulates default/prepay probabilities based on basic risk factors.
        """
        # Base probabilities
        p_def = pl.lit(0.05)
        p_prep = pl.lit(0.10)
        
        if "interest_rate" in df.columns:
            # Higher rate -> higher default, lower prepay
            p_def += (pl.col("interest_rate") - 0.05) * 0.5
            p_prep -= (pl.col("interest_rate") - 0.05) * 1.5
            
        if "current_ltv" in df.columns:
            # Higher LTV -> higher default
            p_def += (pl.col("current_ltv") - 0.80) * 0.1
            
        if "dti" in df.columns:
            # Higher DTI -> higher default
            p_def += (pl.col("dti") - 36.0) / 100.0 * 0.2
            
        if "credit_score" in df.columns:
            # Lower score -> higher default
            p_def += (700.0 - pl.col("credit_score")) / 1000.0 * 0.15

        if "macro_stress_index" in df.columns:
            p_def += pl.col("macro_stress_index") * 0.05
            p_prep -= pl.col("macro_stress_index") * 0.02

        # Clamp probabilities
        return df.with_columns([
            p_def.clip(lower_bound=0.001, upper_bound=0.999).alias("prob_default"),
            p_prep.clip(lower_bound=0.001, upper_bound=0.999).alias("prob_prepay")
        ])

    def run(
        self, 
        df: pl.DataFrame, 
        predict_fn: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None
    ):
        """
        Executes the full scenario intelligence pipeline.
        """
        logger.info("==================================================")
        logger.info("STARTING SCENARIO & STRESS PIPELINE")
        logger.info("==================================================")

        predict = predict_fn or self._default_proxy_model
        
        scenarios = ["Base", "Adverse-Credit", "High-Prepayment"]
        mc_results_raw = {}
        mc_summaries = {}

        # 1. Monte Carlo Simulations
        for sc in scenarios:
            res_matrix = self.simulator.run_simulation(df, sc, predict)
            mc_results_raw[sc] = res_matrix
            mc_summaries[sc] = self.simulator.get_confidence_intervals(res_matrix)
            
        # 2. Segment Vulnerability Analysis
        # Score the base and adverse scenarios deterministically (without stochastic noise)
        base_shocked = self.mapper.apply_macro_shock(df, "Base")
        adv_shocked = self.mapper.apply_macro_shock(df, "Adverse-Credit")
        
        base_scores = predict(base_shocked)
        adv_scores = predict(adv_shocked)
        
        vulnerability_df = self.stress_analyzer.analyze_vulnerability(df, base_scores, adv_scores)
        top_vulnerable = self.stress_analyzer.get_top_vulnerable_segments(vulnerability_df)

        # 3. Tornado Sensitivity
        sensitivities = self.tornado_engine.compute_sensitivities(df, predict)
        self.tornado_engine.generate_tornado_chart(
            sensitivities, 
            output_path=os.path.join(self.figures_dir, "tornado_sensitivity.html")
        )

        # 4. Generate Visualizations & Outputs
        self._plot_distributions(mc_results_raw)
        self._plot_segment_heatmap(vulnerability_df)
        
        self.narrator.generate_report(
            mc_summaries, 
            top_vulnerable, 
            sensitivities,
            output_path=os.path.join(self.reports_dir, "SCENARIO_STRESS_REPORT.md")
        )
        
        # Save raw MC results to parquet for dashboard
        self._save_results(mc_results_raw)

        logger.info("==================================================")
        logger.info("SCENARIO PIPELINE COMPLETE")
        logger.info("==================================================")

    def _plot_distributions(self, mc_results_raw: Dict[str, np.ndarray]):
        """Plots overlaid density distributions of default rates."""
        import pandas as pd
        
        all_data = []
        for sc, matrix in mc_results_raw.items():
            # matrix[:, 0] is default rate
            df_temp = pd.DataFrame({"Default Rate": matrix[:, 0] * 100, "Scenario": sc})
            all_data.append(df_temp)
            
        if not all_data:
            return
            
        plot_df = pd.concat(all_data)
        
        fig = px.histogram(
            plot_df, x="Default Rate", color="Scenario", 
            marginal="box", barmode="overlay",
            title="Monte Carlo Portfolio Default Distributions (1,000 Runs)",
            histnorm='probability density'
        )
        fig.update_layout(template="plotly_white")
        
        out_path = os.path.join(self.figures_dir, "scenario_comparison_distributions.html")
        fig.write_html(out_path)
        logger.info(f"Distribution plot saved to {out_path}")

    def _plot_segment_heatmap(self, vulnerability_df: pl.DataFrame):
        """Generates a heatmap of credit band vs vintage vulnerability."""
        if vulnerability_df.height == 0:
            return
            
        # We need to filter and reshape if both dimensions are present in the raw df
        # But vulnerability_df gives 1D slices. We will just plot a bar chart of top 15.
        pdf = vulnerability_df.head(15).to_pandas()
        pdf["Segment"] = pdf["dimension"] + ": " + pdf["segment_value"]
        
        fig = px.bar(
            pdf, x="relative_risk_delta", y="Segment", orientation="h",
            color="relative_risk_delta", color_continuous_scale="Reds",
            title="Top 15 Most Vulnerable Portfolio Segments (Adverse-Credit Delta)"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_white")
        
        out_path = os.path.join(self.figures_dir, "segment_vulnerability_heatmap.html")
        fig.write_html(out_path)
        logger.info(f"Vulnerability heatmap (bar) saved to {out_path}")

    def _save_results(self, mc_results_raw: Dict[str, np.ndarray]):
        import pyarrow as pa
        
        records = []
        for sc, matrix in mc_results_raw.items():
            for i in range(matrix.shape[0]):
                records.append({
                    "scenario": sc,
                    "iteration": i,
                    "default_rate": float(matrix[i, 0]),
                    "cpr": float(matrix[i, 1]),
                    "loss_severity": float(matrix[i, 2])
                })
        
        df = pl.DataFrame(records)
        out_path = os.path.join(self.data_dir, "simulation_results.parquet")
        df.write_parquet(out_path)
        logger.info(f"Simulation results persisted to {out_path}")
