"""
Unsupervised Anomaly Detection Stack — ml_detectors.py

Implements two complementary unsupervised models:
    1. Isolation Forest — partitions feature space; loans requiring fewer
       splits to isolate are anomalies (low contamination = ~3% expected outliers).
    2. Tabular Autoencoder (PyTorch) — learns the manifold of normal loan behavior;
       records with high MSE reconstruction error deviate from learned norms.

Both models score each loan in [0, 1] via percentile ranking, then are fused
by AnomalyFusionEngine into the final 0–100 composite anomaly score.
"""

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from typing import List, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ===========================================================================
# PyTorch Tabular Autoencoder
# ===========================================================================

class TabularAutoencoder(nn.Module):
    """
    Symmetric bottleneck autoencoder for tabular loan features.

    Architecture: D → D/2 → D/4 → D/2 → D
    Trained on healthy (non-flagged) loan records; anomalies are detected
    by their elevated mean-squared reconstruction error.

    The hidden dimension D/4 forces the network to learn a compressed
    representation of normal loan behavior patterns.
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Number of input features (D).
        """
        super().__init__()
        h1 = max(input_dim // 2, 8)
        h2 = max(input_dim // 4, 4)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes per-record MSE reconstruction error.

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            Tensor of shape (N,): Per-sample reconstruction loss.
        """
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=1)


# ===========================================================================
# Unified Unsupervised Stack
# ===========================================================================

class UnsupervisedAnomalyStack:
    """
    Trains Isolation Forest and Tabular Autoencoder; returns per-record
    anomaly scores scaled to [0, 1] via robust percentile ranking.
    """

    def __init__(
        self,
        contamination: float = 0.03,
        n_estimators: int = 200,
        ae_epochs: int = 30,
        ae_batch_size: int = 512,
        ae_lr: float = 1e-3,
        random_state: int = 42,
    ):
        """
        Args:
            contamination: Expected proportion of anomalies (Isolation Forest).
            n_estimators: Number of trees in Isolation Forest.
            ae_epochs: Training epochs for the autoencoder.
            ae_batch_size: Mini-batch size for autoencoder training.
            ae_lr: Learning rate for autoencoder Adam optimizer.
            random_state: Reproducibility seed.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.ae_epochs = ae_epochs
        self.ae_batch_size = ae_batch_size
        self.ae_lr = ae_lr
        self.random_state = random_state

        self.scaler = RobustScaler()
        self.iso_forest: Optional[IsolationForest] = None
        self.autoencoder: Optional[TabularAutoencoder] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Percentile boundaries fitted on training data for score normalization
        self._iso_p0: float = 0.0
        self._iso_p100: float = 1.0
        self._ae_p0: float = 0.0
        self._ae_p100: float = 1.0

        logger.info(
            f"UnsupervisedAnomalyStack initialized. "
            f"Device={self.device}, contamination={contamination}"
        )

    # ------------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------------

    def fit_isolation_forest(
        self, X: np.ndarray
    ) -> "UnsupervisedAnomalyStack":
        """
        Fits the Isolation Forest on scaled continuous features.

        Raw scores are inverted (sklearn returns negative scores for outliers)
        so that higher values correspond to greater abnormality.

        Args:
            X: Numpy array of continuous features, shape (N, D).

        Returns:
            Self (for chaining).
        """
        logger.info(f"Fitting Isolation Forest on {X.shape[0]:,} records...")
        X_scaled = self.scaler.fit_transform(X)

        self.iso_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.iso_forest.fit(X_scaled)

        # Fit percentile boundaries on training set for normalization
        raw_scores = -self.iso_forest.score_samples(X_scaled)  # invert: higher = more anomalous
        self._iso_p0 = float(np.percentile(raw_scores, 0))
        self._iso_p100 = float(np.percentile(raw_scores, 100))
        logger.info("Isolation Forest training complete.")
        return self

    def score_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        """
        Scores records with the fitted Isolation Forest.

        Returns:
            np.ndarray: Percentile-normalized anomaly scores in [0, 1].
        """
        if self.iso_forest is None:
            raise RuntimeError("Isolation Forest not fitted. Call fit_isolation_forest() first.")
        X_scaled = self.scaler.transform(X)
        raw = -self.iso_forest.score_samples(X_scaled)
        return self._percentile_normalize(raw, self._iso_p0, self._iso_p100)

    # ------------------------------------------------------------------
    # Tabular Autoencoder
    # ------------------------------------------------------------------

    def fit_autoencoder(
        self, X: np.ndarray
    ) -> "UnsupervisedAnomalyStack":
        """
        Trains the Tabular Autoencoder using MSE loss on scaled features.

        The model is trained only on the training population. Reconstruction
        error on held-out or test records reveals distributional anomalies.

        Args:
            X: Numpy array of continuous features, shape (N, D).

        Returns:
            Self (for chaining).
        """
        logger.info(f"Training Tabular Autoencoder on {X.shape[0]:,} records...")
        X_scaled = self.scaler.transform(X).astype(np.float32)

        dataset = TensorDataset(torch.tensor(X_scaled))
        loader = DataLoader(dataset, batch_size=self.ae_batch_size, shuffle=True, drop_last=False)

        self.autoencoder = TabularAutoencoder(input_dim=X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.ae_lr)
        criterion = nn.MSELoss()

        self.autoencoder.train()
        for epoch in range(1, self.ae_epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.autoencoder(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"  AE Epoch {epoch:03d}/{self.ae_epochs}: loss={epoch_loss / len(loader):.6f}")

        # Compute training reconstruction error bounds for normalization
        train_errors = self._compute_ae_errors(X_scaled)
        self._ae_p0 = float(np.percentile(train_errors, 0))
        self._ae_p100 = float(np.percentile(train_errors, 100))
        logger.info("Autoencoder training complete.")
        return self

    def score_autoencoder(self, X: np.ndarray) -> np.ndarray:
        """
        Scores records using reconstruction error from the fitted autoencoder.

        Returns:
            np.ndarray: Percentile-normalized reconstruction errors in [0, 1].
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not fitted. Call fit_autoencoder() first.")
        X_scaled = self.scaler.transform(X).astype(np.float32)
        errors = self._compute_ae_errors(X_scaled)
        return self._percentile_normalize(errors, self._ae_p0, self._ae_p100)

    # ------------------------------------------------------------------
    # Unified Fit / Score
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "UnsupervisedAnomalyStack":
        """Fits both Isolation Forest and Autoencoder sequentially."""
        self.fit_isolation_forest(X)
        self.fit_autoencoder(X)
        return self

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns normalized scores from both models.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (iso_scores, ae_scores), each in [0, 1].
        """
        return self.score_isolation_forest(X), self.score_autoencoder(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ae_errors(self, X_scaled: np.ndarray) -> np.ndarray:
        """Computes per-record reconstruction error on a scaled numpy array."""
        self.autoencoder.eval()
        tensor = torch.tensor(X_scaled).to(self.device)
        errors = self.autoencoder.reconstruction_error(tensor)
        return errors.cpu().numpy()

    @staticmethod
    def _percentile_normalize(
        arr: np.ndarray, p0: float, p100: float
    ) -> np.ndarray:
        """
        Min-max normalizes using training-set percentile bounds.
        Clips output to [0, 1] to handle test-set values outside training range.
        """
        denom = max(p100 - p0, 1e-9)
        return np.clip((arr - p0) / denom, 0.0, 1.0)

    @staticmethod
    def select_numeric_features(df: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts all non-null numeric columns from a Polars DataFrame for ML scoring.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (numpy array, list of column names used).
        """
        numeric_cols = [
            c for c in df.columns
            if df.schema[c] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            and df[c].null_count() < df.height  # at least some values present
        ]
        X = df.select(numeric_cols).fill_null(0).to_numpy().astype(np.float32)
        return X, numeric_cols
