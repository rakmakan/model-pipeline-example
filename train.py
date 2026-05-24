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
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_pipeline.reader import FEATURES, load_train
from model_pipeline.base_model import BaseModel
from model_pipeline.config import load_and_validate
from model_pipeline.preprocessor import Preprocessor

MODEL_REGISTRY_PATH = Path("model_registry.json")


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
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)

    if version in registry["models"]:
        raise ValueError(f"Model version {version} already exists in registry.")

    registry["models"][version] = {
        "version": version,
        "status": "candidate",
        "data_version": data_version,
        "artifact_path": str(artifact_path),
        "eval_report_path": None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "promoted_at": None,
        "promotion_gate": None,
    }

    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def main():
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
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)
    if model_version in registry["models"]:
        raise SystemExit(f"Model version {model_version} already exists in registry. Choose a different --model-version.")

    X_train, y_train = load_train(data_version)
    fraud_rate = y_train.mean()
    print(f"Training data: {len(X_train)} rows, {fraud_rate:.3%} fraud")

    hparams = config["model"]["hyperparameters"]
    model = LogisticRegressionModel(hparams)
    model.fit(X_train, y_train)

    artifact_path = Path(f"models/{model_version}")
    config_snapshot = dict(config)
    config_snapshot["_data_version"] = data_version
    config_snapshot["_model_version"] = model_version
    _pack_artifact(model, artifact_path, data_version, config_snapshot)
    _register_candidate(model_version, data_version, artifact_path)

    print(f"Packed artifact → {artifact_path}")
    print(f"Registered {model_version} as candidate in model_registry.json")
    print(f"Next step: python validate.py --model-version {model_version}")


if __name__ == "__main__":
    main()
