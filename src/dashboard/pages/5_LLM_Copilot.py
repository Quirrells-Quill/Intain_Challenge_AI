"""
5_LLM_Copilot.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import os

st.set_page_config(layout="wide", page_title="LLM Copilot")
st.title("🤖 LLM Reviewer Copilot")

col_chat, col_audit = st.columns([2, 1])

with col_chat:
    st.subheader("Verification Chat")
    st.markdown("Ask natural language questions regarding data dictionaries, loan anomalies, or verification rules.")
    
    st.warning("⚖️ **COMPLIANCE NOTICE:** LLM output is a **recommendation, not a final underwriting decision**. All outputs must be verified by a human analyst.")
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am the Intain Verification Agent AI Copilot. How can I assist you with today's pool?"}
        ]

    messages_container = st.container(height=400)
    for message in st.session_state.messages:
        # User is 👤, assistant is ⚡
        avatar = "👤" if message["role"] == "user" else "⚡"
        messages_container.chat_message(message["role"], avatar=avatar).write(message["content"])
        
    if prompt := st.chat_input("Ask about a rule, a specific loan, or a feature..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        messages_container.chat_message("user", avatar="👤").write(prompt)
        
        try:
            import google.generativeai as genai
            # Initialize with the user-provided key
            genai.configure(api_key="AIzaSyAAjPLFZHkxBnvdtL5unu92Ly2aSyNEVfk")
            # Strict update to requested model per prompt instructions
            model = genai.GenerativeModel('models/gemini-3.6-flash')
            
            with st.spinner("Analyzing..."):
                response = model.generate_content(
                    f"You are a strict, professional AI Underwriting Assistant for Intain. Answer this: {prompt}"
                )
                
            reply = response.text
            st.session_state.messages.append({"role": "assistant", "content": reply})
            messages_container.chat_message("assistant", avatar="⚡").write(reply)
            
        except Exception as e:
            err_msg = f"API Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            messages_container.chat_message("assistant", avatar="⚡").write(err_msg)

with col_audit:
    st.subheader("Live Compliance Ledger")
    st.markdown("Immutable record of AI interactions.")
    db_path = "reports/llm_audit_trail.db"
    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                audit_df = pd.read_sql("SELECT timestamp, loan_id, guardrail_status FROM audit_logs ORDER BY timestamp DESC LIMIT 10", conn)
            st.dataframe(audit_df, hide_index=True)
        except Exception:
            st.warning("Audit DB exists but is currently locked or unreadable.")
    else:
        st.warning("Audit DB not found. Run Stage 8 Copilot to generate logs.")
