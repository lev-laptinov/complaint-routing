"""The reference window of a model version, stored next to the model in its MLflow run.

    uv run python -m complaint_routing.monitoring.reference --alias champion   # backfill v1

Reference = the validation months (2026-03 -> 04, the last labelled months before the test
period), scored by that model version, with the same columns the routing service logs per
request (`score_frame`). Not the whole training set: it spans 2022-2026 and two taxonomy changes,
so any recent window would "drift" from it on every run. The validation window is what the model
was selected on, it has the model's own predictions (needed for prediction drift) and labels.

train.py logs it for every new candidate; this command backfills a version trained before Lab 4.
The artifact is `monitoring/reference.parquet` in the version's run; the drift job reads it for
whatever version the alias points to, so promoting a model also switches its reference.
"""

import argparse
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from complaint_routing import config
from complaint_routing.features import mix_columns
from complaint_routing.serving.features import describe_text, text_analyzer, text_vocabulary

ARTIFACT_DIR = "monitoring"
ARTIFACT = f"{ARTIFACT_DIR}/reference.parquet"
REQUEST_COLUMNS = ["complaint_id", "date_received", "company", "state", "submitted_via", "tags"]


def score_frame(model, df: pd.DataFrame, queues: list[str]) -> pd.DataFrame:
    """Rows as the prediction log stores them: request fields, mix, text columns, prediction."""
    df = df.reset_index(drop=True)
    proba = model.predict_proba(df)
    classes = np.array([str(c) for c in model.classes_])
    out = df[REQUEST_COLUMNS + mix_columns(queues)].copy()
    out = out.join(describe_text(df["narrative"], text_vocabulary(model), text_analyzer(model)))
    out["predicted_queue"] = classes[proba.argmax(axis=1)]
    out["max_proba"] = proba.max(axis=1)
    if "queue" in df:
        out["queue"] = df["queue"].to_numpy()
    return out


def log_reference(
    model, val: pd.DataFrame, queues: list[str], run_id: str | None = None
) -> pd.DataFrame:
    """Score the validation rows and store them as the reference of a run (default: the active
    one). A finished run gets the artifact without being reopened."""
    ref = score_frame(model, val, queues)
    window = "..".join(config.VAL_MONTHS)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "reference.parquet"
        ref.to_parquet(path, index=False)
        if run_id is None:
            mlflow.log_artifact(str(path), artifact_path=ARTIFACT_DIR)
            mlflow.set_tag("reference_window", window)
        else:
            MlflowClient().log_artifact(run_id, str(path), artifact_path=ARTIFACT_DIR)
            MlflowClient().set_tag(run_id, "reference_window", window)
    return ref


def load_reference(run_id: str, client: MlflowClient | None = None) -> pd.DataFrame:
    client = client or MlflowClient()
    with tempfile.TemporaryDirectory() as d:
        return pd.read_parquet(client.download_artifacts(run_id, ARTIFACT, d))


def validation_rows() -> pd.DataFrame:
    from complaint_routing.train import load_dataset, split  # heavy: rebuilds from DVC files

    return split(load_dataset())["val"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="complaint-router")
    p.add_argument("--alias", default="champion")
    args = p.parse_args()
    from complaint_routing.registry import resolve

    r = resolve(args.model, args.alias)
    model = mlflow.sklearn.load_model(r.uri)
    queues = config.queues()
    ref = log_reference(model, validation_rows(), queues, run_id=r.run_id)
    MlflowClient().set_model_version_tag(
        args.model, r.version, "reference_window", "..".join(config.VAL_MONTHS)
    )
    print(f"reference for {args.model} v{r.version}: {len(ref)} rows -> run {r.run_id}/{ARTIFACT}")
    print(ref["predicted_queue"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
