"""
Contribution Visualizer — contribution_viz.py

Generates interactive Plotly charts tracking LLM error modes and adoption rates.
"""

import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

class ContributionVisualizer:
    """
    Renders Agentic ML statistics into interactive HTML figures.
    """

    def __init__(self, output_dir: str = "reports/figures/governance"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_charts(self, metrics: Dict[str, Any]):
        """Generates all required governance visualizations."""
        if not metrics:
            return
            
        self._plot_adoption_pie(metrics["status_counts"])
        self._plot_error_bar(metrics["error_counts"])
        self._plot_timeline(metrics["raw_ledger"])

    def _plot_adoption_pie(self, status_counts: Dict[str, int]):
        """Donut chart of ACCEPTED vs REJECTED vs MODIFIED outputs."""
        labels = list(status_counts.keys())
        values = list(status_counts.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.4,
            marker_colors=['#2ca02c', '#ff7f0e', '#d62728'] # Green, Orange, Red mapped appropriately if ordered
        )])
        
        fig.update_layout(title_text="AI Prompt Output Adoption Rate", template="plotly_white")
        
        out_path = os.path.join(self.output_dir, "ai_adoption_pie.html")
        fig.write_html(out_path)

    def _plot_error_bar(self, error_counts: Dict[str, int]):
        """Horizontal bar chart quantifying types of LLM errors."""
        if not error_counts:
            return
            
        labels = list(error_counts.keys())
        values = list(error_counts.values())
        
        fig = px.bar(
            x=values, y=labels, orientation='h',
            title="LLM Error Modes Encountered & Corrected",
            labels={'x': 'Frequency', 'y': 'Error Category'},
            color=values, color_continuous_scale="Reds"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_white")
        
        out_path = os.path.join(self.output_dir, "error_mode_bar.html")
        fig.write_html(out_path)

    def _plot_timeline(self, raw_ledger: list):
        """Chronological step chart mapping development stages."""
        df = pd.DataFrame(raw_ledger)
        df['step'] = range(1, len(df) + 1)
        
        color_map = {"ACCEPTED": "#2ca02c", "MODIFIED": "#ff7f0e", "REJECTED": "#d62728"}
        
        fig = px.scatter(
            df, x="step", y="phase", color="output_status",
            title="AI Development Chronology & Intervention Map",
            labels={"step": "Prompt Sequence", "phase": "Development Stage"},
            hover_data=["prompt"],
            color_discrete_map=color_map,
            size_max=15
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        fig.update_layout(template="plotly_white", yaxis={'categoryorder': 'array', 'categoryarray': df['phase'].unique()})
        
        out_path = os.path.join(self.output_dir, "development_timeline.html")
        fig.write_html(out_path)
