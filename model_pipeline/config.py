import json
from pathlib import Path

VALID_MODEL_TYPES = {"LogisticRegression"}
VALID_PREPROCESSOR_TYPES = {"StandardScaler"}

DEFAULT_CONFIG_PATH = Path("config.json")


def load_and_validate(path: Path | None = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    with open(path) as f:
        config = json.load(f)

    errors = []

    model_type = config.get("model", {}).get("type")
    if model_type not in VALID_MODEL_TYPES:
        errors.append(f"model.type must be one of {VALID_MODEL_TYPES}, got: {model_type!r}")

    preprocessor_type = config.get("preprocessor", {}).get("type")
    if preprocessor_type not in VALID_PREPROCESSOR_TYPES:
        errors.append(f"preprocessor.type must be one of {VALID_PREPROCESSOR_TYPES}, got: {preprocessor_type!r}")

    training = config.get("training", {})
    if "target_recall" not in training:
        errors.append("training.target_recall is required")
    elif not (0.0 < training["target_recall"] <= 1.0):
        errors.append(f"training.target_recall must be in (0, 1], got: {training['target_recall']}")

    if "split_seed" not in training:
        errors.append("training.split_seed is required")

    hparams = config.get("model", {}).get("hyperparameters", {})
    if "max_iter" in hparams and not isinstance(hparams["max_iter"], int):
        errors.append("model.hyperparameters.max_iter must be an integer")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return config
