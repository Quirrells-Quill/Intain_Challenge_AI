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
from src.dashboard.components import get_dynamic_ai_takeaway

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
    
    with st.expander("🤖 AI Drift Assessment (Live Generation)"):
        prompt = f"Assess this PSI Drift Data. Features: {features}. Corresponding PSI values: {psi_values}. Note any feature > 0.2 as significant drift requiring retraining."
        st.write(get_dynamic_ai_takeaway(prompt))

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
    
    with st.expander("🤖 AI Error Assessment (Live Generation)"):
        prompt = f"Assess this False Positive vs False Negative ratio on a mortgage default model. True Negatives: {z[0][0]}, False Positives: {z[0][1]}, False Negatives: {z[1][0]}, True Positives: {z[1][1]}. Explain why False Positives are generally less dangerous than False Negatives in credit risk."
        st.write(get_dynamic_ai_takeaway(prompt))
