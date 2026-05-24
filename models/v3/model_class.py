import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from base_model import BaseModel
from preprocessor import Preprocessor
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(BaseModel):
    def __init__(self, hyperparameters: dict):
        self.preprocessor = Preprocessor()
        self.model = LogisticRegression(**hyperparameters)
        self._hyperparameters = hyperparameters

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(X)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.preprocessor.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.preprocess(X))[:, 1]

    def save(self, path: Path) -> None:
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        self.preprocessor.save(path)

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionModel":
        obj = cls.__new__(cls)
        with open(path / "model.pkl", "rb") as f:
            obj.model = pickle.load(f)
        obj.preprocessor = Preprocessor.load(path)
        obj._hyperparameters = {}
        return obj
