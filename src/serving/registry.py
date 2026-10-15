"""MLflow model registry with champion / challenger aliases.

The API never loads a model by file path: it asks the registry for whatever version currently holds the
`champion` alias, so promoting a challenger is an alias move, not a redeploy.

    python -m src.serving.registry            # register model_v1 as champion, model_v2 as challenger
"""
import json
import os

import lightgbm as lgb
import mlflow
from mlflow import MlflowClient

from src.data import PROCESSED

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "fraud-lgbm"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def register(model_file: str, meta_file: str, alias: str, label: str) -> str:
    """Log a LightGBM model + its threshold as a new registry version and point `alias` at it."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("fraud-week3-registry")
    meta = json.loads(open(meta_file).read())
    booster = lgb.Booster(model_file=model_file)
    with mlflow.start_run(run_name=f"register_{label}"):
        mlflow.log_params({k: v for k, v in meta.get("params", {}).items()})
        mlflow.log_metrics({"threshold": meta["threshold"], "n_iter": meta["n_iter"]})
        info = mlflow.lightgbm.log_model(booster, name="model", registered_model_name=MODEL_NAME)
        mlflow.log_artifact(meta_file, artifact_path="model_meta")
    client = _client()
    version = str(info.registered_model_version)
    client.set_model_version_tag(MODEL_NAME, version, "threshold", repr(meta["threshold"]))
    client.set_model_version_tag(MODEL_NAME, version, "label", label)
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    return version


def load_alias(alias: str = "champion", name: str = MODEL_NAME):
    """Return (booster, threshold, version) for the registry version currently holding `alias`."""
    client = _client()
    mv = client.get_model_version_by_alias(name, alias)
    booster = mlflow.lightgbm.load_model(f"models:/{name}@{alias}")
    return booster, float(mv.tags["threshold"]), mv.version


def alias_version(alias: str, name: str = MODEL_NAME) -> str | None:
    """Version currently behind `alias` as a str, or None if the alias is not set (cheap: no model download)."""
    try:
        return str(_client().get_model_version_by_alias(name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def promote(alias_from: str = "challenger", alias_to: str = "champion", name: str = MODEL_NAME) -> str:
    """Move `alias_to` onto the version that holds `alias_from` (the previous holder keeps no alias)."""
    client = _client()
    version = client.get_model_version_by_alias(name, alias_from).version
    client.set_registered_model_alias(name, alias_to, version)
    return version


def main():
    v1 = register(str(PROCESSED / "model_v1.txt"), str(PROCESSED / "model_v1.json"), "champion", "tuned_v1")
    v2 = register(str(PROCESSED / "model_v2.txt"), str(PROCESSED / "model_v2.json"), "challenger",
                  "wide_search_v2")
    print(f"champion -> version {v1} (model_v1), challenger -> version {v2} (model_v2)")


if __name__ == "__main__":
    main()
