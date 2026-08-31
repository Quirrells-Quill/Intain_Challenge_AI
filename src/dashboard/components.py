"""
Streamlit Reusable UI Widgets — components.py
"""

import streamlit as st
import polars as pl
import os
import numpy as np

@st.cache_data
def load_data_safe(path: str, fallback_rows: int = 500) -> pl.DataFrame:
    """Aggressively cached safe data loader ensuring UI latency < 500ms."""
    if os.path.exists(path):
        return pl.read_parquet(path)
    
    # Graceful fallback for UI display if parquets haven't been compiled
    st.sidebar.warning(f"Data missing: {path}. Using mock view. Run `run_all.py` to generate.")
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        'loan_id': [f"L_TEST_{i}" for i in range(fallback_rows)],
        'reporting_month': ["2026-01-01"] * fallback_rows,
        'current_balance': rng.uniform(50000, 500000, fallback_rows),
        'interest_rate': rng.uniform(0.03, 0.09, fallback_rows),
        'prob_12m_default': rng.uniform(0.01, 0.4, fallback_rows),
        'prob_12m_prepay': rng.uniform(0.05, 0.5, fallback_rows),
        'anomaly_score': rng.integers(0, 100, fallback_rows),
        'exception_required': rng.choice([True, False], p=[0.15, 0.85], size=fallback_rows),
        'top_drivers': ["RULE_1; dti" for _ in range(fallback_rows)],
        'recommended_action': rng.choice(["Auto-Approve", "Manual Triage", "Reject/Repurchase"], fallback_rows),
        'state': rng.choice(["CA", "NY", "TX", "FL", "IL"], fallback_rows),
        'vintage': rng.choice([2019, 2020, 2021, 2022], fallback_rows)
    })

def load_custom_css():
    st.markdown("""
        <style>
        /* Fix cut-off header issue */
        [data-testid="stAppViewBlockContainer"] {
            padding-top: 2rem !important;
            margin-top: 0 !important;
        }
        /* Institutional Metric Cards */
        [data-testid="stMetric"] {
            background-color: #111111;
            border: 1px solid #333333;
            padding: 15px;
            border-radius: 4px;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Courier New', monospace !important;
            color: #00E676 !important;
            text-shadow: 0px 0px 8px rgba(0, 230, 118, 0.4);
        }
        /* Make expanders sleek */
        .streamlit-expanderHeader {
            font-family: 'Courier New', monospace;
            color: #00E676 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_metric_card(title: str, value: str, delta: str = None):
    """Renders a standardized metric card."""
    st.metric(label=title, value=value, delta=delta)

def render_status_banner(action: str):
    """Renders a high-contrast institutional status banner."""
    if action == "Auto-Approve":
        st.markdown('<div style="background-color:#0b2e13; border: 1px solid #00E676; padding: 12px; border-radius: 4px; color: #00E676; font-weight: bold; font-family: monospace; text-align: center; text-transform: uppercase; letter-spacing: 2px;">🟢 VERIFICATION STATUS: AUTO-APPROVE</div>', unsafe_allow_html=True)
    elif action == "Manual Triage":
        st.markdown('<div style="background-color:#332b00; border: 1px solid #ffc107; padding: 12px; border-radius: 4px; color: #ffc107; font-weight: bold; font-family: monospace; text-align: center; text-transform: uppercase; letter-spacing: 2px;">🟡 VERIFICATION STATUS: MANUAL TRIAGE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color:#3b0f12; border: 1px solid #d62728; padding: 12px; border-radius: 4px; color: #ff5252; font-weight: bold; font-family: monospace; text-align: center; text-transform: uppercase; letter-spacing: 2px;">🔴 VERIFICATION STATUS: REJECT / FLAG</div>', unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def get_dynamic_ai_takeaway(prompt: str) -> str:
    """Generates a dynamic, live AI takeaway using the provided Gemini API key."""
    try:
        import google.generativeai as genai
        genai.configure(api_key="AIzaSyAAjPLFZHkxBnvdtL5unu92Ly2aSyNEVfk")
        model = genai.GenerativeModel('models/gemini-3.6-flash')
        # Instruct the model to be a hyper-professional quant analyst
        sys_prompt = "You are a quantitative risk analyst for a hedge fund. Analyze the following data in exactly 2 concise, professional sentences. Do not use filler words."
        response = model.generate_content(f"{sys_prompt}\n\nData: {prompt}")
        return response.text
    except Exception as e:
        return f"AI Analyst Module Temporarily Offline. (Error: {str(e)})"
