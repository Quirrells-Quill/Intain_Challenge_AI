"""
Scenario Mapper — scenario_mapper.py

Translates macroeconomic factor shocks (Base, Adverse-Credit, High-Prepayment)
into micro loan-level feature adjustments using vectorized operations.
"""

import polars as pl
from pathlib import Path
from typing import Dict, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MacroScenarioMapper:
    """
    Applies macroeconomic shocks to loan-level features.
    
    Reads baseline scenario definitions from configs/macro_scenarios.csv.
    """

    def __init__(self, scenarios_path: str = "configs/macro_scenarios.csv"):
        self.scenarios_path = Path(scenarios_path)
        self.scenarios: Dict[str, Dict[str, float]] = {}
        self._load_scenarios()
        logger.info(f"MacroScenarioMapper initialized with scenarios: {list(self.scenarios.keys())}")

    def _load_scenarios(self):
        """Loads scenarios from CSV."""
        if not self.scenarios_path.exists():
            raise FileNotFoundError(f"Scenario configuration not found: {self.scenarios_path}")
        
        df = pl.read_csv(str(self.scenarios_path))
        for row in df.iter_rows(named=True):
            name = row.pop("scenario_name")
            self.scenarios[name] = {k: float(v) for k, v in row.items()}

    def get_scenario_params(self, scenario_name: str) -> Dict[str, float]:
        """Retrieves raw parameters for a given scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {list(self.scenarios.keys())}")
        return self.scenarios[scenario_name]

    def apply_macro_shock(
        self,
        df: pl.DataFrame,
        scenario_name: str,
        shock_scale: float = 1.0,
        custom_shocks: Optional[Dict[str, float]] = None
    ) -> pl.DataFrame:
        """
        Applies a defined macroeconomic shock to the loan features.

        Transformations:
            - Interest Rate: Direct basis point addition (e.g., +150 bps = +0.015).
            - LTV: Adjusted by HPI shift. LTV_new = Current Balance / (Original Value * (1 + delta HPI)).
            - DTI: Inflated by unemployment proxy shift.

        Args:
            df: Loan-level feature DataFrame.
            scenario_name: 'Base', 'Adverse-Credit', or 'High-Prepayment'.
            shock_scale: Multiplier for the scenario severity (1.0 = standard).
            custom_shocks: Overrides for specific stochastic iterations.

        Returns:
            pl.DataFrame: Shocked feature matrix.
        """
        params = self.get_scenario_params(scenario_name).copy()
        if custom_shocks:
            params.update(custom_shocks)

        rate_bps_change = params.get("interest_rate_bps_change", 0.0) * shock_scale
        unemp_pct_change = params.get("unemployment_pct_change", 0.0) * shock_scale
        hpi_pct_change = params.get("hpi_pct_change", 0.0) * shock_scale
        dti_pct_change = params.get("dti_pct_change", 0.0) * shock_scale

        logger.debug(f"Applying '{scenario_name}' shock (scale={shock_scale:.2f})")

        out_cols = []
        
        # 1. Interest Rate adjustment (assuming current rate is in decimal, e.g., 0.05 for 5%)
        if "interest_rate" in df.columns:
            rate_delta = rate_bps_change / 10000.0
            out_cols.append(
                (pl.col("interest_rate") + rate_delta).clip(lower_bound=0.005, upper_bound=0.30).alias("interest_rate")
            )
        
        # 2. LTV adjustment via HPI (House Price Index)
        if "current_ltv" in df.columns:
            hpi_multiplier = 1.0 + (hpi_pct_change / 100.0)
            # Prevent division by zero or extreme negative property values
            hpi_multiplier = max(hpi_multiplier, 0.1) 
            
            # LTV = Bal / Value. New LTV = Bal / (Value * hpi_multiplier) = LTV / hpi_multiplier
            out_cols.append(
                (pl.col("current_ltv") / hpi_multiplier).clip(lower_bound=0.0, upper_bound=2.0).alias("current_ltv")
            )

        # 3. DTI adjustment
        if "dti" in df.columns:
            # Assuming DTI is continuous [0, 100]
            out_cols.append(
                (pl.col("dti") + dti_pct_change).clip(lower_bound=0.0, upper_bound=100.0).alias("dti")
            )
            
        # 4. Synthesize a macro stress index for models that use it directly
        # Higher index = more economic stress
        stress_index = (unemp_pct_change * 2.0) + (rate_bps_change / 50.0) - hpi_pct_change
        out_cols.append(pl.lit(stress_index).alias("macro_stress_index"))

        if out_cols:
            return df.with_columns(out_cols)
        return df
