from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model. Implementations must call self.preprocess() internally."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probability for each row. Shape: (n,)"""

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features. Called inside fit() and predict_proba(). Never called directly by consumers."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model weights and preprocessor state to artifact directory."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load a fully ready model from an artifact directory."""
