import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = self._scaler.fit_transform(X)
        self._fitted = True
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_transform first.")
        scaled = self._scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def save(self, path: Path) -> None:
        with open(path / "preprocessor.pkl", "wb") as f:
            pickle.dump(self._scaler, f)

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        obj = cls.__new__(cls)
        with open(path / "preprocessor.pkl", "rb") as f:
            obj._scaler = pickle.load(f)
        obj._fitted = True
        return obj
