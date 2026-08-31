import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
"""
6_Drift_Error_Analysis.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide", page_title="Drift & Error Analysis")
st.title("📈 Drift & Error Analysis")

st.markdown("""
This dashboard monitors the temporal stability of our dataset (Train vs. OOT Validation) 
and provides a granular breakdown of model errors (False Positives vs. False Negatives).
""")

col_drift, col_error = st.columns(2)

with col_drift:
    st.subheader("Data Drift: Population Stability Index (PSI)")
    st.markdown("Comparing feature distributions between the 2024 training window and the 2025 out-of-time (OOT) validation window.")
    
    # Mock data for PSI drift
    features = ["credit_score", "dti", "interest_rate", "current_balance", "loan_age"]
    psi_values = [0.02, 0.08, 0.22, 0.05, 0.11]
    
    colors = ['#2ca02c' if psi < 0.1 else '#ff7f0e' if psi < 0.2 else '#d62728' for psi in psi_values]
    
    fig_drift = go.Figure(data=[go.Bar(
        x=features,
        y=psi_values,
        marker_color=colors
    )])
    
    fig_drift.add_hline(y=0.1, line_dash="dash", line_color="green", annotation_text="Safe (<0.1)")
    fig_drift.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Action Required (>0.2)")
    
    fig_drift.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Feature Drift (PSI)", 
        yaxis_title="PSI Score",
        margin=dict(t=30, b=0, l=0, r=0)
    )
    fig_drift.update_yaxes(showgrid=True, gridcolor='#222222')
    st.plotly_chart(fig_drift, use_container_width=True)
    
    if max(psi_values) > 0.2:
        st.warning(f"**Drift Alert:** 'interest_rate' exhibits significant drift (PSI > 0.2). This is expected due to the macroeconomic rate hikes in 2025.")

with col_error:
    st.subheader("Error Analysis: Default Prediction (12m)")
    st.markdown("Confusion Matrix evaluated on the strictly segmented OOT Validation set.")
    
    # Mock Confusion Matrix data
    # [True Negative, False Positive]
    # [False Negative, True Positive]
    z = [[8500, 420],
         [ 180, 900]]
    
    x = ['Predicted: Current/Prepay', 'Predicted: Default']
    y = ['Actual: Current/Prepay', 'Actual: Default']
    
    # Heatmap
    fig_cm = px.imshow(z, x=x, y=y, text_auto=True, aspect="auto", color_continuous_scale='Tealrose')
    fig_cm.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Confusion Matrix (Out-of-Time Validation)",
        margin=dict(t=30, b=0, l=0, r=0)
    )
    fig_cm.update_traces(textfont=dict(family='monospace', size=16))
    st.plotly_chart(fig_cm, use_container_width=True)
    
    st.info("""
    **Error Insights:**
    * **False Positives (420):** Model predicted default, but loan survived. Often driven by borrowers with high DTI who aggressively prioritized mortgage payments over revolving credit.
    * **False Negatives (180):** Model missed the default. Driven predominantly by sudden, unobservable income shocks (e.g., localized tech layoffs not yet reflected in macro unemployment data).
    """)

