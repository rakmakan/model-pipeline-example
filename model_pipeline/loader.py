import importlib.util
import json
import sys
from pathlib import Path

from model_pipeline.base_model import BaseModel


def load_model(version: str) -> BaseModel:
    artifact_path = Path(f"models/{version}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}. Run train.py first.")

    metadata = json.loads((artifact_path / "metadata.json").read_text())
    class_name = metadata["model_class_name"]

    sys.path.insert(0, str(artifact_path))
    try:
        spec = importlib.util.spec_from_file_location(
            f"artifact_{version}_model_class", artifact_path / "model_class.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ModelClass = getattr(module, class_name)
    finally:
        sys.path.pop(0)

    model = ModelClass.load(artifact_path)
    model._metadata = metadata
    return model


def get_threshold(version: str) -> float:
    artifact_path = Path(f"models/{version}")
    metadata = json.loads((artifact_path / "metadata.json").read_text())
    threshold = metadata.get("threshold")
    if threshold is None:
        raise ValueError(f"Model {version} has no threshold set. Run validate.py first.")
    return threshold


def get_features(version: str) -> list[str]:
    artifact_path = Path(f"models/{version}")
    metadata = json.loads((artifact_path / "metadata.json").read_text())
    return metadata["features"]
