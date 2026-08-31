"""
Counterfactual Explanation Engine — counterfactuals.py

Generates actionable minimal-distance counterfactuals to rescue high-risk loans.
"""

import numpy as np
import polars as pl
from typing import Callable, Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CounterfactualEngine:
    """
    Finds the minimal actionable feature perturbation required to transition
    a loan from 'High Risk' to 'Approved'.
    """

    def __init__(self, predict_fn: Callable[[pl.DataFrame], pl.DataFrame], target_col: str = "prob_default", threshold: float = 0.10):
        self.predict_fn = predict_fn
        self.target_col = target_col
        self.threshold = threshold
        
        # Actionability bounds definition
        self.actionable_features = {
            "current_balance": {"direction": "decrease", "step": 1000.0, "max_change": 0.5}, # Up to 50% paydown
            "credit_score": {"direction": "increase", "step": 10.0, "max_change": 100.0}, # Up to 100 point increase
            "interest_rate": {"direction": "decrease", "step": 0.0025, "max_change": 0.02}, # Up to 200 bps cut
            "dti": {"direction": "decrease", "step": 2.0, "max_change": 10.0} # Up to 10% DTI improvement
        }

    def _normalize_distance(self, orig_val: float, new_val: float, feature: str) -> float:
        """Computes a rough normalized L1 distance penalty."""
        diff = abs(new_val - orig_val)
        if feature == "current_balance":
            return diff / max(orig_val, 1.0) # percentage of balance
        elif feature == "credit_score":
            return diff / 100.0
        elif feature == "interest_rate":
            return diff / 0.02
        elif feature == "dti":
            return diff / 10.0
        return diff

    def generate_counterfactual_prescription(self, loan_record: pl.DataFrame) -> Dict[str, Any]:
        """
        Greedy search to find a minimal-distance valid counterfactual.
        
        Args:
            loan_record: Single-row DataFrame of the high-risk loan.
            
        Returns:
            Dictionary containing the prescription and distance metric.
        """
        orig_prob = self.predict_fn(loan_record)[self.target_col][0]
        
        if orig_prob <= self.threshold:
            return {"status": "Already below threshold", "orig_prob": orig_prob}

        current_record = loan_record.clone()
        adjustments = {}
        total_distance = 0.0
        
        # We perform a greedy iterative search over actionable features
        for _ in range(20): # Max iterations
            best_move = None
            best_prob = 1.0
            best_dist_increase = float('inf')
            
            for feature, bounds in self.actionable_features.items():
                if feature not in current_record.columns:
                    continue
                    
                orig_val = loan_record[feature][0]
                curr_val = current_record[feature][0]
                
                # Propose a step
                step = bounds["step"]
                if bounds["direction"] == "decrease":
                    proposed_val = curr_val - step
                    # Check max change bounds
                    if orig_val - proposed_val > orig_val * bounds["max_change"] if feature == "current_balance" else bounds["max_change"]:
                        continue
                else:
                    proposed_val = curr_val + step
                    if proposed_val - orig_val > bounds["max_change"]:
                        continue
                        
                # Evaluate step
                test_record = current_record.with_columns(pl.lit(proposed_val).alias(feature))
                test_prob = self.predict_fn(test_record)[self.target_col][0]
                
                if test_prob < orig_prob:
                    dist_increase = self._normalize_distance(curr_val, proposed_val, feature)
                    # We want the highest prob drop per unit of distance
                    efficiency = (current_record - test_prob) / max(dist_increase, 1e-6) if type(current_record) == float else (orig_prob - test_prob) / max(dist_increase, 1e-6)
                    
                    if test_prob < best_prob:
                        best_prob = test_prob
                        best_move = (feature, proposed_val, dist_increase)
                        
            if not best_move:
                break # Local optima reached
                
            feature, new_val, dist = best_move
            current_record = current_record.with_columns(pl.lit(new_val).alias(feature))
            adjustments[feature] = new_val
            total_distance += dist
            
            if best_prob <= self.threshold:
                break
                
        final_prob = self.predict_fn(current_record)[self.target_col][0]
        success = final_prob <= self.threshold
        
        prescription = []
        for feat, val in adjustments.items():
            orig_v = loan_record[feat][0]
            if feat == "current_balance":
                prescription.append(f"Reduce balance by ${orig_v - val:,.0f}")
            elif feat == "credit_score":
                prescription.append(f"Increase credit score by {val - orig_v:.0f} points")
            elif feat == "interest_rate":
                prescription.append(f"Apply {(orig_v - val)*10000:.0f} bps rate concession")
            elif feat == "dti":
                prescription.append(f"Decrease DTI by {orig_v - val:.1f}%")
                
        return {
            "success": success,
            "orig_prob": float(orig_prob),
            "final_prob": float(final_prob),
            "total_distance": float(total_distance),
            "adjustments": adjustments,
            "prescription": prescription
        }
