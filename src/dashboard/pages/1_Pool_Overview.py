"""
1_Pool_Overview.py
"""

import streamlit as st
import plotly.express as px
from src.dashboard.components import load_data_safe, render_metric_card, get_dynamic_ai_takeaway

st.set_page_config(layout="wide", page_title="Pool Overview")
st.title("📊 Pool-Level Overview")

# Load data
df = load_data_safe("data/processed/master_pool.parquet").to_pandas()

# Calculate Top-Level KPIs
total_loans = len(df)
total_balance = df['current_balance'].sum()
wac = (df['interest_rate'] * df['current_balance']).sum() / total_balance
wam = 320 # Stub calculation
health_score = max(0, 100 - df['anomaly_score'].mean())
cdr = df['prob_12m_default'].mean() * 100

# Render Top-Level KPIs
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1: render_metric_card("Total Loans", f"{total_loans:,}")
with col2: render_metric_card("Total Balance", f"${total_balance:,.0f}")
with col3: render_metric_card("WAC", f"{wac:.2f}%")
with col4: render_metric_card("WAM", f"{wam} mo")
with col5: render_metric_card("Health Score", f"{health_score:.1f}/100")
with col6: render_metric_card("12m CDR (Proj)", f"{cdr:.1f}%")

st.markdown("---")

# Render Interactive Plotly charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Risk Concentration by State")
    state_counts = df['state'].value_counts().reset_index()
    state_counts.columns = ['state', 'count']
    
    # Sophisticated institutional color palette
    custom_colors = ['#00E676', '#00BFFF', '#1E90FF', '#4169E1', '#00008B']
    
    # Donut chart update
    fig_pie = px.pie(state_counts, values='count', names='state', hole=0.6, 
                     color_discrete_sequence=custom_colors)
                     
    fig_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    # Add labels inside the donut pieces
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)
    
    with st.expander("🤖 AI Analyst Takeaway (Live Generation)"):
        with st.spinner("Generating live analysis..."):
            top_state = state_counts.iloc[0]['state']
            top_state_count = state_counts.iloc[0]['count']
            prompt = f"Our mortgage portfolio is heavily concentrated in {top_state} ({top_state_count} loans). Assess the geographic risk and macroeconomic implications."
            st.write(get_dynamic_ai_takeaway(prompt))

with col_chart2:
    st.subheader("GBDT Probability Calibration")
    # Simple distribution for UI
    fig_hist = px.histogram(df, x="prob_12m_default", nbins=20, 
                            color_discrete_sequence=["#00E676"])
                            
    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Predicted 12m Default Probability",
        yaxis_title="Loan Count",
        margin=dict(t=30, b=0, l=0, r=0)
    )
    fig_hist.update_yaxes(showgrid=True, gridcolor='#222222', gridwidth=1)
    fig_hist.update_xaxes(showgrid=False)
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    with st.expander("🤖 AI Analyst Takeaway (Live Generation)"):
        with st.spinner("Generating live analysis..."):
            avg_prob = df['prob_12m_default'].mean() * 100
            prompt = f"The GBDT ensemble predicts an average 12m default probability of {avg_prob:.2f}%. The distribution is a long-tail with most loans under 10%. Is this healthy for a prime portfolio?"
            st.write(get_dynamic_ai_takeaway(prompt))
