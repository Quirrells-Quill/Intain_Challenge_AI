"""
Time-Aware Cross-Validation Splitter for Loan Portfolios

Implements chronological splitting that strictly prevents any loan_id from
bleeding across train/validation boundaries — a hard requirement in any
financial time series modeling context.
"""

import polars as pl
import numpy as np
from typing import Iterator, Tuple, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesLoanSplitter:
    """
    Custom cross-validation splitter for loan performance data.

    Enforces chronological ordering and loan-level boundary integrity.
    A single loan_id will NEVER appear in both train and validation folds.

    This prevents the model from memorizing loan-specific trajectories
    rather than learning generalized credit behavior.
    """

    def __init__(self, n_splits: int = 5, gap_months: int = 1):
        """
        Args:
            n_splits (int): Number of temporal folds to create.
            gap_months (int): Number of months gap between train end and val start.
                              Prevents look-ahead bias from reporting lags.
        """
        self.n_splits = n_splits
        self.gap_months = gap_months

    def split(
        self, df: pl.DataFrame, date_col: str = "reporting_month"
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates (train_indices, val_indices) with a mandatory 12-month blackout buffer.

        PILLAR 3 COMPLIANCE (SYSTEM_STANDARD.md):
            Train Period : [T_start            ->  T_train_cutoff]
            Blackout     : (T_train_cutoff      ->  T_train_cutoff + 12 months]  <- DROPPED
            OOT Period   : [T_train_cutoff + 12 months  ->  T_eval_end]

        The 12-month blackout prevents next_12m_default_flag labels from contaminating
        training features. gap_months is overridden to a minimum of 12.

        Args:
            df (pl.DataFrame): Full feature matrix with date_col present.
            date_col (str): Column representing the time dimension.

        Yields:
            Tuple[np.ndarray, np.ndarray]: Train and validation index arrays.
        """
        unique_months = sorted(df[date_col].unique().to_list())
        total_months = len(unique_months)

        # Enforce 12-month minimum blackout regardless of gap_months setting
        effective_gap = max(self.gap_months, 12)
        fold_size = max(1, (total_months - effective_gap) // (self.n_splits + 1))

        logger.info(
            f"TimeSeriesLoanSplitter [PILLAR 3]: {total_months} months, "
            f"{self.n_splits} folds, fold_size≈{fold_size}m, "
            f"blackout={effective_gap}m (12-month leakage guard active)"
        )

        for fold in range(self.n_splits):
            train_end_idx = fold_size * (fold + 1)
            # Mandatory 12-month blackout — this slice is never used for training or validation
            val_start_idx = train_end_idx + effective_gap
            val_end_idx = val_start_idx + fold_size

            if val_end_idx > total_months:
                logger.warning(f"Fold {fold + 1}: Insufficient months after blackout. Skipping.")
                break

            train_months = unique_months[:train_end_idx]
            blackout_months = unique_months[train_end_idx:val_start_idx]  # auditable, never touched
            val_months = unique_months[val_start_idx:val_end_idx]

            # Strict loan-level boundary: any loan_id in val is purged from train
            val_loan_ids = set(
                df.filter(pl.col(date_col).is_in(val_months))["loan_id"].to_list()
            )

            train_mask = df[date_col].is_in(train_months) & ~df["loan_id"].is_in(val_loan_ids)
            val_mask = df[date_col].is_in(val_months)

            train_indices = np.where(train_mask.to_numpy())[0]
            val_indices = np.where(val_mask.to_numpy())[0]

            logger.info(
                f"Fold {fold + 1}: train={len(train_indices):,}, "
                f"blackout={len(blackout_months)}m (dropped), "
                f"val={len(val_indices):,} | "
                f"train_end={train_months[-1]}, val_start={val_months[0]}"
            )

            yield train_indices, val_indices

    def get_final_test_split(
        self, df: pl.DataFrame, test_ratio: float = 0.2, date_col: str = "reporting_month"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a definitive holdout split for final model evaluation.
        The last `test_ratio` of the time axis forms the holdout set.

        Args:
            df (pl.DataFrame): Full feature matrix.
            test_ratio (float): Proportion of tail months used for holdout.
            date_col (str): Temporal column name.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Train and test index arrays.
        """
        unique_months = sorted(df[date_col].unique().to_list())
        cutoff_idx = int(len(unique_months) * (1 - test_ratio))
        train_months = unique_months[:cutoff_idx]
        test_months = unique_months[cutoff_idx:]

        test_loan_ids = set(
            df.filter(pl.col(date_col).is_in(test_months))["loan_id"].to_list()
        )

        train_mask = df[date_col].is_in(train_months) & ~df["loan_id"].is_in(test_loan_ids)
        test_mask = df[date_col].is_in(test_months)

        return (
            np.where(train_mask.to_numpy())[0],
            np.where(test_mask.to_numpy())[0],
        )
