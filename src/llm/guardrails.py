"""
Guardrails — guardrails.py

Programmatic hallucination detector and numerical verifier.
Verifies probabilities and rule violations against ground-truth ML outputs.
"""

import re
from typing import Dict, Any, Tuple
from src.utils.logger import get_logger
from src.llm.schemas import ReviewerSummarySchema

logger = get_logger(__name__)


class HallucinationGuardrail:
    """
    Independent programmatic guardrail intercepting LLM outputs.
    """

    def __init__(self, prob_tolerance: float = 0.03):
        self.prob_tolerance = prob_tolerance

    def _extract_decimals_and_percentages(self, text: str) -> list[float]:
        """Extracts percentages (like 18.2%) and raw decimals (like 0.182)."""
        results = []
        # Match percentages like 18.2%, 45%
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        for p in pct_matches:
            results.append(float(p) / 100.0)
            
        # Match standalone decimals < 1.0 (assuming probabilities)
        dec_matches = re.findall(r'\b0\.\d+\b', text)
        for d in dec_matches:
            results.append(float(d))
            
        return results

    def validate_numerical_consistency(
        self, 
        llm_output: ReviewerSummarySchema, 
        actual_ml_outputs: Dict[str, Any]
    ) -> Tuple[bool, str, ReviewerSummarySchema]:
        """
        Assertion 1: If LLM narrative quotes a probability, verify |p_claimed - p_actual| <= 0.03.
        Assertion 2: If LLM claims an exception, verify it is in the ML validation matrix.
        
        Returns:
            (is_valid, status, validated_or_fallback_output)
        """
        combined_text = llm_output.summary + " " + llm_output.reviewer_notes
        
        # 1. Probability Check
        actual_prob = float(actual_ml_outputs.get("prob_default", 0.0))
        claimed_probs = self._extract_decimals_and_percentages(combined_text)
        
        for cp in claimed_probs:
            if abs(cp - actual_prob) > self.prob_tolerance:
                logger.warning(f"Hallucination detected: Claimed prob {cp:.3f}, Actual {actual_prob:.3f}")
                return self._reject_and_fallback(actual_ml_outputs, f"Probability mismatch: {cp:.3%} vs {actual_prob:.3%}")

        # 2. Rule Exception Check
        # If the LLM mentions "RULE_", check if it actually exists in anomalies
        rule_mentions = set(re.findall(r'RULE_\w+', combined_text.upper()))
        actual_rules = set()
        
        if "anomaly_data" in actual_ml_outputs:
            anom = actual_ml_outputs["anomaly_data"]
            if isinstance(anom, dict):
                # E.g. top_drivers: ["RULE_1", "RULE_2"]
                drivers = anom.get("top_drivers", [])
                actual_rules.update([d for d in drivers if str(d).startswith("RULE_")])
                
        for rm in rule_mentions:
            if rm not in actual_rules:
                logger.warning(f"Hallucination detected: Claimed rule '{rm}' not in actual drivers {actual_rules}.")
                return self._reject_and_fallback(actual_ml_outputs, f"Fabricated rule violation: {rm}")

        # Passed
        return True, "PASSED", llm_output

    def _reject_and_fallback(self, actual_ml_outputs: Dict[str, Any], reason: str) -> Tuple[bool, str, ReviewerSummarySchema]:
        """
        Rejection Handler: Generates a deterministic template-based note.
        """
        p = actual_ml_outputs.get('prob_default', 0.0)
        s = actual_ml_outputs.get('anomaly_data', {}).get('anomaly_score', 0)
        
        fallback_summary = f"DETERMINISTIC FALLBACK: Loan has a default probability of {p:.1%} and an anomaly score of {s}."
        
        fallback_output = ReviewerSummarySchema(
            summary=fallback_summary,
            risk_assessment="HIGH" if p > 0.3 else ("MEDIUM" if p > 0.1 else "LOW"),
            key_drivers=[],
            recommended_action="Manual Triage",
            reviewer_notes=f"WARNING: LLM output rejected due to hallucination ({reason}). Proceed with manual review.",
            grounding_citations=[],
            confidence_score=0.0,
            disclaimer="Deterministic Fallback Activated."
        )
        
        return False, "REJECTED_HALLUCINATION", fallback_output
