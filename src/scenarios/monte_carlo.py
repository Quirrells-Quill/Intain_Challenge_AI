"""
Monte Carlo Simulation Engine — monte_carlo.py

Vectorized 1,000-run stochastic simulation engine.
Introduces multivariate Gaussian noise to macroeconomic parameters and evaluates
portfolio-level risk distributions under stress conditions.
"""

import numpy as np
import polars as pl
from typing import Dict, Tuple, Callable, List
from src.scenarios.scenario_mapper import MacroScenarioMapper
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MonteCarloSimulator:
    """
    Executes stochastic Monte Carlo simulations to derive empirical
    confidence bounds for portfolio risk metrics.
    """

    def __init__(
        self,
        mapper: MacroScenarioMapper,
        n_simulations: int = 1000,
        random_seed: int = 42,
    ):
        self.mapper = mapper
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        logger.info(f"MonteCarloSimulator initialized with {n_simulations} iterations.")

    def run_simulation(
        self,
        df: pl.DataFrame,
        scenario_name: str,
        predict_fn: Callable[[pl.DataFrame], pl.DataFrame],
    ) -> np.ndarray:
        """
        Runs the Monte Carlo simulation for a given scenario.

        Args:
            df: Baseline loan-level feature DataFrame.
            scenario_name: Scenario configuration name.
            predict_fn: A function that takes a feature DataFrame and returns
                        a DataFrame containing 'prob_default' and 'prob_prepay'.

        Returns:
            np.ndarray: Array of shape (n_simulations, 3) containing:
                [Portfolio Default Rate, Portfolio CPR, Loss Severity]
        """
        logger.info(f"Starting Monte Carlo for scenario: '{scenario_name}' ({self.n_simulations} runs)")

        base_params = self.mapper.get_scenario_params(scenario_name)
        
        # Define stochastic volatility around the base macro parameters
        # Example: +/- 50 bps on rate, +/- 0.5% on unemployment, +/- 2.0% on HPI
        stdevs = {
            "interest_rate_bps_change": 50.0,
            "unemployment_pct_change": 0.5,
            "hpi_pct_change": 2.0,
            "dti_pct_change": 1.0,
        }

        # Generate [1000, 4] matrix of noise using independent Gaussians for simplicity
        # (A full covariance matrix Sigma could be injected here)
        noise_matrix = {
            k: self.rng.normal(loc=0.0, scale=std, size=self.n_simulations)
            for k, std in stdevs.items()
        }

        # Validate that required columns exist for aggregation
        has_balance = "current_balance" in df.columns

        results = np.zeros((self.n_simulations, 3), dtype=np.float32)

        for i in range(self.n_simulations):
            # Construct iteration-specific shocks
            custom_shocks = {
                k: base_params.get(k, 0.0) + noise_matrix[k][i]
                for k in stdevs.keys()
            }

            # 1. Apply shocks (vectorized over loans)
            shocked_df = self.mapper.apply_macro_shock(df, scenario_name, custom_shocks=custom_shocks)

            # 2. Score portfolio (vectorized over loans)
            # predict_fn should return a DF with 'prob_default', 'prob_prepay'
            scored_df = predict_fn(shocked_df)

            # 3. Aggregate metrics
            p_default = scored_df["prob_default"].to_numpy()
            p_prepay = scored_df["prob_prepay"].to_numpy()
            
            port_default_rate = float(p_default.mean())
            port_cpr = float(p_prepay.mean())
            
            if has_balance:
                bals = df["current_balance"].to_numpy()
                # Assuming 40% Loss Given Default (LGD)
                loss_severity = float((p_default * bals * 0.40).sum() / max(bals.sum(), 1.0))
            else:
                loss_severity = port_default_rate * 0.40

            results[i, 0] = port_default_rate
            results[i, 1] = port_cpr
            results[i, 2] = loss_severity

            if (i + 1) % 200 == 0:
                logger.debug(f"  Simulation progress: {i + 1}/{self.n_simulations} runs complete.")

        return results

    def get_confidence_intervals(self, results_matrix: np.ndarray, confidence: float = 0.90) -> Dict[str, Dict[str, float]]:
        """
        Calculates Point Estimate (Mean), P5, P50, and P95 from empirical simulation arrays.
        
        Args:
            results_matrix: Shape (n_simulations, 3) from run_simulation.
            confidence: Alpha level (default 0.90 gives P5 and P95).
            
        Returns:
            Dict mapping metric names to their CI dictionaries.
        """
        metrics = ["default_rate", "cpr", "loss_severity"]
        lower_pct = (1.0 - confidence) / 2.0 * 100
        upper_pct = (1.0 + confidence) / 2.0 * 100

        summary = {}
        for idx, metric in enumerate(metrics):
            arr = results_matrix[:, idx]
            summary[metric] = {
                "mean": float(np.mean(arr)),
                "p_lower": float(np.percentile(arr, lower_pct)),
                "p50": float(np.median(arr)),
                "p_upper": float(np.percentile(arr, upper_pct)),
            }
        return summary
