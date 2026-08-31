"""
Smoke Test for LLM Copilot — test_llm_smoke.py
"""

import os
import unittest
import polars as pl
from unittest.mock import patch

from src.llm.schemas import ReviewerSummarySchema
from src.llm.copilot_service import ReviewerCopilotService
from src.llm.curated_failures import compile_failure_report

class TestLLMCopilot(unittest.TestCase):

    def setUp(self):
        os.makedirs("data", exist_ok=True)
        os.makedirs("configs", exist_ok=True)
        with open("data/data_dictionary.md", "w") as f:
            f.write("## DTI\nDebt-to-Income ratio.")
        with open("configs/validation_rules.json", "w") as f:
            f.write('{"rules": {"rule_1": "balance must be > 0"}}')
            
        self.copilot = ReviewerCopilotService(model_name="mock-model")
        self.copilot.client.mock_mode = True

    def test_copilot_clean_run(self):
        loan_data = {"balance": 250000.0, "DTI": 35}
        ml_outputs = {"prob_default": 0.182, "anomaly_data": {"anomaly_score": 0}}
        
        # In mock mode, gemini returns a mock object
        result = self.copilot.generate_loan_dossier_note("L123", loan_data, ml_outputs)
        
        self.assertIsNotNone(result)
        # The mock summary says "18.2%", which perfectly matches ml_outputs, so guardrail passes.
        self.assertTrue(result.confidence_score > 0)

    def test_copilot_hallucination_catch(self):
        loan_data = {"balance": 250000.0}
        # Force a mismatch: ML says 0.50, Mock LLM says 18.2%
        ml_outputs = {"prob_default": 0.500, "anomaly_data": {"anomaly_score": 0}}
        
        result = self.copilot.generate_loan_dossier_note("L456", loan_data, ml_outputs)
        
        self.assertIsNotNone(result)
        # Guardrail should have caught the mismatch
        self.assertEqual(result.confidence_score, 0.0)
        self.assertEqual(result.recommended_action, "Manual Triage")
        self.assertIn("DETERMINISTIC FALLBACK", result.summary)

    def test_curated_failures(self):
        compile_failure_report("reports/test_failures.md")
        self.assertTrue(os.path.exists("reports/test_failures.md"))
        
    def test_audit_export(self):
        df = self.copilot.audit.export_audit_summary()
        self.assertIsInstance(df, pl.DataFrame)

if __name__ == "__main__":
    unittest.main()
