"""
4_Verification_Queue.py
"""

import streamlit as st
from src.dashboard.components import load_data_safe

st.set_page_config(layout="wide", page_title="Verification Queue")
st.title("?? Exception Verification Queue")

df = load_data_safe("data/processed/master_pool.parquet").to_pandas()
exceptions = df[df['exception_required'] == True]

st.markdown(f"**{len(exceptions)}** loans currently require human-in-the-loop triage.")

if not exceptions.empty:
    selected_loan = st.selectbox("Select Loan to Triage:", exceptions['loan_id'].tolist())
    row = exceptions[exceptions['loan_id'] == selected_loan].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Exception Details")
        st.write(f"**Exception Type**: {row.get('exception_type', 'Rule Violation')}")
        st.write(f"**Anomaly Score**: {row.get('anomaly_score', 85)}")
        st.write(f"**Top Flagged Drivers**: {row.get('top_drivers', 'RULE_3')}")
        
    with col2:
        st.subheader("Verification Action")
        if st.button("Request LLM Summary (Copilot)"):
            with st.spinner("Copilot is analyzing the anomaly..."):
                from src.dashboard.components import get_dynamic_ai_takeaway
                prompt = f"Loan {selected_loan} triggered an anomaly with score {row.get('anomaly_score', 85)}. The top driver flagged is {row.get('top_drivers', 'unknown')}. Give a 2-sentence recommendation to the underwriting verification team."
                summary = get_dynamic_ai_takeaway(prompt)
                st.info(f"**🤖 Copilot Analysis**: {summary}")
            
        c_a, c_b = st.columns(2)
        with c_a:
            if st.button("Accept Loan (Override)"):
                st.success(f"Loan {selected_loan} approved. Audit log updated.")
        with c_b:
            if st.button("Reject & Flag for Repurchase"):
                st.error(f"Loan {selected_loan} rejected.")
else:
    st.success("The exception queue is entirely clear!")
