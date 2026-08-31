"""
Feature Engineering & Feature Store Module

Generates a highly predictive feature matrix for structured finance asset pools.
Leverages fast Polars expressions for rolling windows, macro-interactions, and 
ratio-based financial features. Implements a Parquet-backed Feature Store.
"""

import polars as pl
from pathlib import Path
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineeringPipeline:
    """
    Constructs deterministically engineered features without data leakage.
    Produces a time-aligned, feature-rich matrix for predictive modeling and survival analysis.
    """

    def __init__(self):
        logger.info("Initialized FeatureEngineeringPipeline")

    def generate_static_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        1. Static & Ratio Features.
        Calculates critical financial ratios based on static attributes and current snapshots.
        
        Derived Features:
        - current_ltv: Adjusts Loan-to-Value dynamically as principal is paid down.
        - principal_paydown_rate: Percentage of the original balance that has been amortized.
        - loan_age_ratio: The proportion of the loan's life that has elapsed.
        """
        logger.info("Generating Static & Ratio Features...")
        
        if "original_term" not in df.columns:
            # Fallback if original_term is missing (e.g., standard 360 months for 30yr mortgage)
            df = df.with_columns(pl.lit(360.0).alias("original_term"))
            
        return df.with_columns([
            # current_ltv
            ((pl.col("current_balance") / pl.col("original_balance")) * pl.col("original_ltv")).alias("current_ltv"),
            
            # principal_paydown_rate
            (1.0 - (pl.col("current_balance") / pl.col("original_balance"))).alias("principal_paydown_rate"),
            
            # loan_age_ratio
            (pl.col("loan_age") / pl.col("original_term")).alias("loan_age_ratio")
        ])

    def generate_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        2. Temporal & Rolling Window Features.
        Constructs lagging behavior indicators and rolling volatility measures
        using Polars high-performance window functions. 
        
        Calculates:
        - Lagged delinquencies to identify historical defaults (1, 3, 6).
        - Rolling counts of defaults (3, 6, 12 months).
        - Moving average of principal reduction.
        - Exponentially weighted historical delinquency score.
        """
        logger.info("Generating Temporal & Rolling Window Features...")
        
        # Sort strictly by loan_id and time index to prevent target leakage
        df = df.sort(["loan_id", "reporting_month"])
        
        # Binary flag for delinquency (any DPD > 0 means delinquent)
        if "days_past_due" in df.columns:
            df = df.with_columns(
                (pl.col("days_past_due") > 0).cast(pl.Int32).alias("is_delinquent")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("is_delinquent"))
            
        # Calculate balance reduction amount per month
        df = df.with_columns(
            (pl.col("current_balance").shift(1).over("loan_id") - pl.col("current_balance"))
            .alias("balance_reduction")
        )

        # Generate Lags & Rolling Features
        df = df.with_columns([
            # Lags
            pl.col("is_delinquent").shift(1).over("loan_id").alias("delinq_lag_1"),
            pl.col("is_delinquent").shift(3).over("loan_id").alias("delinq_lag_3"),
            pl.col("is_delinquent").shift(6).over("loan_id").alias("delinq_lag_6"),
            
            # Rolling sums (Counts of delinquency occurrences over windows)
            pl.col("is_delinquent").rolling_sum(window_size=3, min_periods=1).over("loan_id").alias("delinq_count_3m"),
            pl.col("is_delinquent").rolling_sum(window_size=6, min_periods=1).over("loan_id").alias("delinq_count_6m"),
            pl.col("is_delinquent").rolling_sum(window_size=12, min_periods=1).over("loan_id").alias("delinq_count_12m"),
            
            # Rolling average balance reduction (volatility tracking)
            pl.col("balance_reduction").rolling_mean(window_size=3, min_periods=1).over("loan_id").alias("bal_reduct_avg_3m"),
            pl.col("balance_reduction").rolling_mean(window_size=6, min_periods=1).over("loan_id").alias("bal_reduct_avg_6m"),
            pl.col("balance_reduction").rolling_mean(window_size=12, min_periods=1).over("loan_id").alias("bal_reduct_avg_12m"),
        ])
        
        # Exponential Time-Decay Weighting on Delinquencies
        # Recent defaults are weighted heavily, decaying exponentially for older defaults
        df = df.with_columns(
            (
                pl.col("is_delinquent").fill_null(0) * 1.0 +
                pl.col("delinq_lag_1").fill_null(0) * 0.8 +
                pl.col("is_delinquent").shift(2).over("loan_id").fill_null(0) * 0.64 + 
                pl.col("delinq_lag_3").fill_null(0) * 0.512
            ).alias("ewm_delinq_score")
        )

        return df

    def generate_macro_interactions(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        3. Macro Proxies & Interactions.
        Embeds economic cyclicality and complex feature interactions.
        
        - Seasonality: extracts month and quarter.
        - Era Tags: classifies origination periods (e.g. high-rate vs low-rate).
        - Cross-features: uncovers nonlinear combinations like Rate * Credit, Age * LTV.
        """
        logger.info("Generating Macro Proxies & Cross-Features...")
        
        if "reporting_month" in df.columns and df.schema["reporting_month"] in [pl.Date, pl.Datetime]:
            df = df.with_columns([
                pl.col("reporting_month").dt.month().alias("seasonality_month"),
                pl.col("reporting_month").dt.quarter().alias("seasonality_quarter")
            ])
            
        if "origination_month" in df.columns and df.schema["origination_month"] in [pl.Date, pl.Datetime]:
            df = df.with_columns(
                pl.col("origination_month").dt.year().alias("vintage_year")
            )
        else:
            df = df.with_columns(pl.lit(2020).alias("vintage_year"))
        
        # Era Tag generation (Proxy: pre-2022 vs post-2022 rate environments)
        df = df.with_columns(
            pl.when(pl.col("vintage_year") >= 2022).then(pl.lit("high-rate-era"))
            .otherwise(pl.lit("low-rate-era"))
            .alias("era_tag")
        )
        
        # Ensure we have numeric proxies for categorical features like credit_score_band
        if "credit_score_band" in df.columns:
            # Extracts the first occurrence of a number in the string (e.g. '700-750' -> 700.0)
            df = df.with_columns(
                pl.col("credit_score_band")
                .cast(pl.Utf8)
                .str.extract(r"(\d+)", 1)
                .cast(pl.Float32)
                .fill_null(700.0)
                .alias("_numeric_credit_proxy")
            )
        else:
            df = df.with_columns(pl.lit(700.0).alias("_numeric_credit_proxy"))
        
        # Complex interactions bridging static risk profiles and dynamic attributes
        df = df.with_columns([
            (pl.col("interest_rate") * pl.col("_numeric_credit_proxy")).alias("cross_rate_x_credit"),
            (pl.col("loan_age") * pl.col("current_ltv")).alias("cross_age_x_ltv")
        ])
        
        # Clean up temporary proxy
        df = df.drop("_numeric_credit_proxy")
        
        return df

    def align_time_series(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        4. Temporal Calendar Alignment.
        Shifts the temporal axis from absolute calendar dates to a relative 'loan_age'.
        This ensures models learn universal behavioral trajectories without overfitting 
        to specific macro-calendar anomalies.
        """
        logger.info("Aligning Time Series to universal loan_age index...")
        
        df = df.sort(["loan_id", "loan_age"])
        
        # Verify alignment by assigning a normalized step integer 
        # (Guarantees monotonic sequence regardless of missed servicer reporting months)
        df = df.with_columns(
            pl.int_range(1, pl.len() + 1).over("loan_id").alias("behavioral_step_index")
        )
        
        return df

    def save_to_feature_store(self, df: pl.DataFrame, output_dir: str = "data/processed/") -> str:
        """
        5. Feature Store Persistence.
        Saves the engineered DataFrame as a partitioned Parquet dataset.
        Partitioning by era_tag enables highly optimal vectorized reads later.
        """
        logger.info(f"Saving to Feature Store at {output_dir}...")
        
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        if "era_tag" in df.columns:
            try:
                # Attempt to write a Hive-partitioned pyarrow dataset
                df.write_parquet(
                    str(path), 
                    use_pyarrow=True, 
                    partition_by=["era_tag"], 
                    compression="snappy"
                )
                logger.info(f"Feature matrix successfully persisted as partitioned dataset at {path}")
                return str(path)
            except Exception as e:
                logger.warning(f"Partitioned save failed (likely missing pyarrow), falling back to single file: {e}")
                
        # Fallback to single monolithic file
        output_path = str(path / "feature_matrix.parquet")
        df.write_parquet(output_path, compression="snappy")
        logger.info(f"Feature matrix successfully persisted to {output_path}")
        
        return output_path

    def run_pipeline(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Executes the full deterministically engineered feature pipeline.
        """
        logger.info("Starting complete Feature Engineering Pipeline...")
        df = self.generate_static_features(df)
        df = self.generate_temporal_features(df)
        df = self.generate_macro_interactions(df)
        df = self.align_time_series(df)
        
        self.save_to_feature_store(df)
        
        logger.info("Pipeline complete. Feature Store updated.")
        return df
