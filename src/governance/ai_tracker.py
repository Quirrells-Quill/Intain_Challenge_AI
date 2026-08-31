"""
AI Contribution Tracker — ai_tracker.py

Parses the AI prompt ledger and evaluates human-AI code sharing metrics.
"""

import json
import os
from collections import Counter
from typing import Dict, Any

class AgenticContributionAnalyzer:
    """
    Ingests AI development ledgers to calculate adoption rates, error frequencies,
    and approximate code-sharing metrics.
    """

    def __init__(self, ledger_path: str = "configs/ai_development_ledger.json"):
        self.ledger_path = ledger_path
        self.ledger_data = self._load_ledger()

    def _load_ledger(self) -> list:
        if not os.path.exists(self.ledger_path):
            raise FileNotFoundError(f"Ledger not found at {self.ledger_path}")
        with open(self.ledger_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def compute_metrics(self) -> Dict[str, Any]:
        """Calculates adoption rate and error mode distributions."""
        total_prompts = len(self.ledger_data)
        if total_prompts == 0:
            return {}

        statuses = [entry.get("output_status") for entry in self.ledger_data]
        status_counts = Counter(statuses)
        
        # Accepted or Modified means AI contribution was heavily utilized
        adopted = status_counts.get("ACCEPTED", 0) + status_counts.get("MODIFIED", 0)
        adoption_rate = adopted / total_prompts

        # Extract Error Modes
        error_modes = [entry.get("error_mode") for entry in self.ledger_data if entry.get("error_mode")]
        error_counts = Counter(error_modes)

        return {
            "total_prompts": total_prompts,
            "status_counts": dict(status_counts),
            "adoption_rate": adoption_rate,
            "error_counts": dict(error_counts),
            "raw_ledger": self.ledger_data
        }

    def compute_code_share(self, src_path: str = "src") -> Dict[str, Any]:
        """
        Estimates total lines of code in the src/ directory and applies a heuristical 
        AI vs Human split (assuming a 90% AI baseline based on typical agentic dev).
        """
        total_lines = 0
        for root, _, files in os.walk(src_path):
            for file in files:
                if file.endswith(".py"):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            total_lines += len(f.readlines())
                    except Exception:
                        pass
        
        # Heuristic calculation: baseline high AI utilization minus penalty for rejections
        metrics = self.compute_metrics()
        adoption_rate = metrics.get("adoption_rate", 0.90)
        
        # E.g., if adoption is 70%, AI share might be ~75% depending on modifications
        ai_share = min(adoption_rate + 0.15, 0.95)
        human_share = 1.0 - ai_share
        
        return {
            "total_lines_of_code": total_lines,
            "ai_generated_lines": int(total_lines * ai_share),
            "human_refactored_lines": int(total_lines * human_share),
            "ai_share_percentage": ai_share * 100,
            "human_share_percentage": human_share * 100
        }
