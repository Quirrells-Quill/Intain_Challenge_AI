"""
Audit Logger — audit_logger.py

SQLite audit log for all prompts, responses, and decisions.
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime
import polars as pl
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AuditTrailLogger:
    """
    Persists LLM interactions to a SQLite database for governance and auditing.
    """

    def __init__(self, db_path: str = "reports/llm_audit_trail.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    loan_id TEXT NOT NULL,
                    prompt_template TEXT,
                    retrieved_context TEXT,
                    model_name TEXT,
                    raw_response TEXT,
                    parsed_json TEXT,
                    guardrail_status TEXT,
                    latency_seconds REAL,
                    human_reviewer_feedback TEXT
                )
            """)
            conn.commit()
        logger.info(f"Audit DB initialized at {self.db_path}")

    def log_interaction(
        self,
        loan_id: str,
        prompt_template: str,
        retrieved_context: str,
        model_name: str,
        raw_response: str,
        parsed_json: str,
        guardrail_status: str,
        latency_seconds: float,
        human_reviewer_feedback: str = "PENDING"
    ):
        """Inserts a new audit record with transactional integrity."""
        log_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_logs 
                    (log_id, timestamp, loan_id, prompt_template, retrieved_context, model_name, raw_response, parsed_json, guardrail_status, latency_seconds, human_reviewer_feedback)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_id, timestamp, loan_id, prompt_template, retrieved_context,
                    model_name, raw_response, parsed_json, guardrail_status,
                    latency_seconds, human_reviewer_feedback
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to write to audit log: {e}")

    def export_audit_summary(self) -> pl.DataFrame:
        """Extracts compliance summaries for hackathon deliverables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Polars natively supports reading from sqlite via fetchall
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM audit_logs")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
            if not rows:
                return pl.DataFrame({c: [] for c in columns})
                
            df = pl.DataFrame(rows, schema=columns)
            return df
        except Exception as e:
            logger.error(f"Failed to export audit summary: {e}")
            return pl.DataFrame()
