"""
Lightweight, picklable scalers.

Why these exist
---------------
The PenSim observation channels span vastly different magnitudes (e.g. vessel
weight ~2.5e5 vs. a ratio near 1). Fitting a GP with unit-lengthscale priors on
raw values is hopeless. We therefore standardize (z-score) every GP input and
output. Doing it *ourselves* (rather than relying on the env's ``normalize``
flag) keeps the world model correct regardless of whether the buffer stored raw
or env-normalized observations.

All scalers are plain numpy and store only mean/std vectors, so they pickle
cleanly and can be reloaded in later files (BCNP, policy learning, rollouts).
"""

import numpy as np


class StandardScaler:
    """Per-dimension z-score scaler: z = (x - mean) / std."""

    def __init__(self, eps: float = 1e-8):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # Guard against zero-variance channels (constant over the buffer).
        std[std < self.eps] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=np.float64) * self.std_ + self.mean_

    def transform_delta(self, dX: np.ndarray) -> np.ndarray:
        """Scale a *difference* of raw values (no mean subtraction)."""
        return np.asarray(dX, dtype=np.float64) / self.std_

    def inverse_transform_delta(self, dZ: np.ndarray) -> np.ndarray:
        """Un-scale a difference back to raw units (no mean addition)."""
        return np.asarray(dZ, dtype=np.float64) * self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def state_dict(self) -> dict:
        return {"mean_": self.mean_, "std_": self.std_, "eps": self.eps}

    @classmethod
    def from_state_dict(cls, d: dict) -> "StandardScaler":
        s = cls(eps=d.get("eps", 1e-8))
        s.mean_ = d["mean_"]
        s.std_ = d["std_"]
        return s
