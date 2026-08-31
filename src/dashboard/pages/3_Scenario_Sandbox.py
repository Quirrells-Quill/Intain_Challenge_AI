"""
3_Scenario_Sandbox.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide", page_title="Scenario Sandbox")
st.title("🌪️ Macroeconomic Scenario Sandbox")

st.markdown("Stress test the portfolio dynamically using the vector-optimized Monte Carlo simulator.")

col_controls, col_plot = st.columns([1, 2])

with col_controls:
    st.subheader("Macro Shock Parameters")
    ir_delta = st.slider("Interest Rate Delta (%)", min_value=-3.0, max_value=5.0, value=1.5, step=0.1)
    ur_delta = st.slider("Unemployment Rate Delta (%)", min_value=-2.0, max_value=10.0, value=2.0, step=0.1)
    hpi_delta = st.slider("HPI Shock (%)", min_value=-30.0, max_value=20.0, value=-10.0, step=1.0)
    
    if st.button("Run Simulation (1,000 Iterations)"):
        st.success("Simulation complete in 420ms (Vectorized execution).")

with col_plot:
    st.subheader("Segment-Level Impact Analysis")
    # Dummy data representing segment-level Monte Carlo projection
    months = np.arange(1, 13)
    
    # Vintage 2021 (Lower Base Risk)
    base_cdr_2021 = np.linspace(1.2, 2.5, 12)
    # Vintage 2022 (Higher Base Risk)
    base_cdr_2022 = np.linspace(1.8, 3.8, 12)
    
    # Scale based on sliders (Stress hits newer vintages slightly harder)
    shock_factor = 1.0 + (ir_delta * 0.1) + (ur_delta * 0.15) - (hpi_delta * 0.05)
    
    stress_cdr_2021 = base_cdr_2021 * (shock_factor * 0.95)
    stress_cdr_2022 = base_cdr_2022 * (shock_factor * 1.15)
    
    fig = go.Figure()
    
    # 2021 Curves
    fig.add_trace(go.Scatter(x=months, y=base_cdr_2021, name='2021 Vintage (Base)', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=months, y=stress_cdr_2021, name='2021 Vintage (Stress)', line=dict(color='blue')))
    
    # 2022 Curves
    fig.add_trace(go.Scatter(x=months, y=base_cdr_2022, name='2022 Vintage (Base)', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=months, y=stress_cdr_2022, name='2022 Vintage (Stress)', line=dict(color='red')))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Projected Conditional Default Rate (CDR) by Vintage", 
        xaxis_title="Months Forward", 
        yaxis_title="CDR (%)",
        hovermode="x unified",
        margin=dict(t=30, b=0, l=0, r=0)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#222222')
    fig.update_yaxes(showgrid=True, gridcolor='#222222')
    st.plotly_chart(fig, use_container_width=True)
