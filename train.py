"""
Trains a fraud detection model, packs it as a self-contained artifact, and registers it as a candidate.

Usage:
    python train.py --data-version v2 --model-version v2
    python train.py --data-version v2 --model-version v2 --config configs/high_recall.json
    python train.py --config models/v2/train_config.json   # reproduce exact run
"""
import argparse
import inspect
import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_pipeline.reader import FEATURES, load_train
from logging_config import setup_logging
from model_pipeline.base_model import BaseModel
from model_pipeline.config import load_and_validate
from model_pipeline.preprocessor import Preprocessor

MODEL_REGISTRY_PATH = Path("model_registry.json")
_EMPTY_MODEL_REGISTRY = {"active": None, "models": {}, "history": []}
logger = logging.getLogger(__name__)


def _load_model_registry() -> dict:
    if not MODEL_REGISTRY_PATH.exists():
        logger.info("model_registry.json not found — initializing empty registry")
        MODEL_REGISTRY_PATH.write_text(json.dumps(_EMPTY_MODEL_REGISTRY, indent=2))
        return {k: v for k, v in _EMPTY_MODEL_REGISTRY.items()}
    return json.loads(MODEL_REGISTRY_PATH.read_text())


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


def _pack_artifact(model: LogisticRegressionModel, artifact_path: Path, data_version: str, config: dict) -> None:
    artifact_path.mkdir(parents=True, exist_ok=True)
    model.save(artifact_path)

    shutil.copy("model_pipeline/base_model.py", artifact_path / "base_model.py")
    shutil.copy("model_pipeline/preprocessor.py", artifact_path / "preprocessor.py")

    class_source = inspect.getsource(LogisticRegressionModel)
    model_class_content = (
        "import pickle\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "from base_model import BaseModel\n"
        "from preprocessor import Preprocessor\n"
        "from sklearn.linear_model import LogisticRegression\n\n\n"
        + class_source
    )
    (artifact_path / "model_class.py").write_text(model_class_content)

    metadata = {
        "model_class_name": "LogisticRegressionModel",
        "schema_version": "v1",
        "data_version": data_version,
        "features": FEATURES,
        "threshold": None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (artifact_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (artifact_path / "train_config.json").write_text(json.dumps(config, indent=2))


def _register_candidate(version: str, data_version: str, artifact_path: Path) -> None:
    registry = _load_model_registry()

    if version in registry["models"]:
        raise ValueError(f"Model version {version} already exists in registry.")

    data_registry = json.loads(Path("data_registry.json").read_text())
    data_hash = data_registry["versions"][data_version]["data_hash"]

    registry["models"][version] = {
        "version": version,
        "status": "candidate",
        "data_version": data_version,
        "data_hash": data_hash,
        "artifact_path": str(artifact_path),
        "eval_report_path": None,
        "replay_metrics": None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "promoted_at": None,
        "promotion_gate": None,
    }

    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", help="Data version from data_registry.json")
    parser.add_argument("--model-version", help="Version name for the new model artifact")
    parser.add_argument("--config", default=None, help="Path to config JSON (defaults to config.json)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None

    if args.config and Path(args.config).name == "train_config.json":
        saved = json.loads(Path(args.config).read_text())
        data_version = saved.get("_data_version", args.data_version)
        model_version = args.model_version or saved.get("_model_version")
        config = load_and_validate(config_path)
    else:
        if not args.data_version or not args.model_version:
            parser.error("--data-version and --model-version are required unless reproducing from train_config.json")
        data_version = args.data_version
        model_version = args.model_version
        config = load_and_validate(config_path)

    # Guard before touching disk — fail fast if version already registered.
    registry = _load_model_registry()
    if model_version in registry["models"]:
        logger.error("Model version %s already exists in registry. Choose a different --model-version.", model_version)
        raise SystemExit(1)

    X_train, y_train = load_train(data_version)
    fraud_rate = y_train.mean()
    logger.info("Training data: %d rows, %.3f%% fraud (data=%s)", len(X_train), fraud_rate * 100, data_version)

    hparams = config["model"]["hyperparameters"]
    logger.debug("Training %s with hyperparameters: %s", config["model"]["type"], hparams)
    model = LogisticRegressionModel(hparams)
    model.fit(X_train, y_train)
    logger.debug("Training complete")

    artifact_path = Path(f"models/{model_version}")
    config_snapshot = dict(config)
    config_snapshot["_data_version"] = data_version
    config_snapshot["_model_version"] = model_version
    _pack_artifact(model, artifact_path, data_version, config_snapshot)
    _register_candidate(model_version, data_version, artifact_path)

    logger.info("Packed artifact → %s", artifact_path)
    logger.info("Registered %s as candidate in model_registry.json", model_version)
    logger.info("Next step: python validate.py --model-version %s", model_version)


if __name__ == "__main__":
    main()
