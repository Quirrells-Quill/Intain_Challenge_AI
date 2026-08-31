"""
Streamlit Main Entry — app.py
"""

import streamlit as st
from src.dashboard.components import load_custom_css

st.set_page_config(
    layout="wide", 
    page_title="Intain-Sight: Verification Agent",
    page_icon="🏦"
)
load_custom_css()

st.title("🏦 Intain-Sight Verification Agent")
st.markdown("### Enterprise Loan Performance Intelligence Engine")

st.markdown("""
Welcome to the Intain-Sight Workspace. 
This dashboard serves as the primary interface for human-in-the-loop verification, macro-scenario stress testing, and explainable AI insights.

**Navigation (Sidebar)**:
- 📊 **1. Pool Overview**: Macro health, WAC/WAM, risk concentration.
- 🔍 **2. Loan Explorer**: Micro loan-level view + SHAP + Counterfactuals.
- 🌪️ **3. Scenario Sandbox**: Segment-level Monte Carlo sliders & Confidence Intervals.
- 🚨 **4. Verification Queue**: Anomaly triage workflow (Accept/Reject/Flag).
- 🤖 **5. LLM Copilot**: RAG-backed chat interface for data queries and governance.
- 📈 **6. Drift & Error Analysis**: Train vs Test drift and False Positive / False Negative tracking.
""")

st.info("👈 Please select a page from the sidebar to begin.")
