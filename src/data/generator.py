"""
High-Fidelity Synthetic Loan Data Generator
Mimics Fannie Mae Single-Family Loan Performance Data properties.
"""

import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
import os

class SyntheticDataGenerator:
    def __init__(self, n_loans: int = 1000, months_per_loan: int = 12, seed: int = 42):
        self.n_loans = n_loans
        self.months_per_loan = months_per_loan
        self.rng = np.random.default_rng(seed)

    def generate(self, output_path: str = "data/raw/loan_monthly_performance.parquet") -> pl.DataFrame:
        print(f"Generating {self.n_loans} high-fidelity synthetic loans...")
        
        # 1. Static Loan Features
        loan_ids = [f"LN_{100000 + i}" for i in range(self.n_loans)]
        vintages = self.rng.choice([2020, 2021, 2022], size=self.n_loans)
        states = self.rng.choice(["CA", "NY", "TX", "FL", "IL"], size=self.n_loans)
        
        # Risk factors
        credit_scores = self.rng.normal(720, 40, size=self.n_loans).clip(500, 850).astype(int)
        dtis = self.rng.normal(35, 10, size=self.n_loans).clip(10, 60)
        interest_rates = 3.0 + (850 - credit_scores) * 0.01 + (dtis - 35) * 0.02 + self.rng.normal(0, 0.5, self.n_loans)
        interest_rates = interest_rates.clip(2.5, 9.5)
        
        orig_balances = self.rng.normal(250000, 75000, size=self.n_loans).clip(50000, 1000000)
        
        # Compute baseline risk (Hidden mathematical relationship for the ML model to learn)
        # Higher DTI and lower Credit Score = Higher Risk
        base_risk = (dtis / 60.0) + (1.0 - (credit_scores / 850.0))
        
        data = []
        start_date = datetime(2023, 1, 1)
        
        for i in range(self.n_loans):
            curr_balance = orig_balances[i]
            is_defaulted = False
            is_prepaid = False
            
            for m in range(self.months_per_loan):
                if is_defaulted or is_prepaid:
                    break
                    
                report_date = start_date + timedelta(days=30 * m)
                
                # Dynamic targets
                # Default hazard increases with base risk
                hazard_default = base_risk[i] * 0.05 + self.rng.uniform(0, 0.02)
                # Prepay hazard increases if interest rates drop (simulated randomly here)
                hazard_prepay = (interest_rates[i] > 6.0) * 0.08 + self.rng.uniform(0, 0.03)
                
                next_12m_default = self.rng.random() < hazard_default
                next_12m_prepay = self.rng.random() < hazard_prepay
                
                data.append({
                    "loan_id": loan_ids[i],
                    "reporting_month": report_date.strftime("%Y-%m-%d"),
                    "vintage": vintages[i],
                    "state": states[i],
                    "credit_score": credit_scores[i],
                    "dti": dtis[i],
                    "interest_rate": interest_rates[i],
                    "current_balance": curr_balance,
                    "target_12m_default": int(next_12m_default),
                    "target_12m_prepay": int(next_12m_prepay)
                })
                
                if next_12m_default:
                    is_defaulted = True
                elif next_12m_prepay:
                    is_prepaid = True
                    
                curr_balance = max(0, curr_balance - self.rng.uniform(500, 1500))

        df = pl.DataFrame(data)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.write_parquet(output_path)
        print(f"Dataset generated with {df.height} rows at {output_path}")
        return df

if __name__ == "__main__":
    gen = SyntheticDataGenerator()
    gen.generate()
