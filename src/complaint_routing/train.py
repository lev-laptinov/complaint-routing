"""Train the baselines and the candidate on the time split; log everything to MLflow.

    uv run python -m complaint_routing.train                 # log 3 runs, register the candidate
    uv run python -m complaint_routing.train --no-register   # e.g. for a reproduction

Every run carries the two lineage tags: data_version (DVC md5 of data/raw) and git_commit.
The dataset is rebuilt from the DVC-versioned raw files with the same clean() and
company_queue_mix() functions the Airflow DAG runs, so a run depends only on (data, commit).
"""

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from complaint_routing import config, lineage
from complaint_routing.baselines import KeywordBaseline, MajorityBaseline
from complaint_routing.features import (
    FEATURE_VERSION,
    LAG_DAYS,
    WINDOW_DAYS,
    company_queue_mix,
    mix_columns,
)
from complaint_routing.transform import clean

MODEL_NAME = "complaint-router"
CATEGORICAL = ["company", "state", "submitted_via", "tags"]
POPULATION_MIX = config.ROOT / "config" / "test_population_mix.csv"


def load_dataset() -> pd.DataFrame:
    mapping, queues = config.load_mapping(), config.queues()
    months = config.months_between(config.FIRST_MONTH, config.LAST_MONTH)
    # Clean month by month, exactly as the DAG does (dedup rules work within a batch).
    df = pd.concat([clean(pd.read_parquet(config.raw_path(m)), mapping)[0] for m in months])
    return company_queue_mix(df, df, queues)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def between(lo_hi: tuple[str, str]) -> pd.DataFrame:
        return df[df["batch_month"].between(*lo_hi)]

    return {"train": between(config.TRAIN_MONTHS), "val": between(config.VAL_MONTHS),
            "test": between(config.TEST_MONTHS)}  # fmt: skip


def build_model(queues: list[str], c: float, max_features: int, min_df: int, seed: int) -> Pipeline:
    features = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=max_features,
                                 sublinear_tf=True, dtype=np.float32), "narrative"),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL),
        ("mix", "passthrough", mix_columns(queues)[:-1]),
    ])  # fmt: skip
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


def scores(y_true, y_pred, queues: list[str], prefix: str) -> dict[str, float]:
    recall = recall_score(y_true, y_pred, labels=queues, average=None, zero_division=0)
    out = {
        f"{prefix}_macro_f1": f1_score(
            y_true, y_pred, labels=queues, average="macro", zero_division=0
        ),
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_min_recall": float(recall.min()),
    }
    out |= {f"{prefix}_recall_{q}": float(r) for q, r in zip(queues, recall, strict=True)}
    if prefix == "test" and POPULATION_MIX.exists():
        # Design 5.4: accuracy if the test months had the full intake mix (incl. credit reporting).
        mix = pd.read_csv(POPULATION_MIX).set_index("queue")["share"]
        out["test_accuracy_reweighted"] = float(
            sum(mix.get(q, 0) * r for q, r in zip(queues, recall, strict=True))
        )
    return out


def lineage_tags() -> dict[str, str]:
    return {
        "data_version": lineage.data_version(),
        "git_commit": lineage.git_commit(),
        "git_dirty": str(lineage.git_dirty()).lower(),
        "feature_version": FEATURE_VERSION,
        "mapping_version": config.mapping_version(),
    }


def log_report(y_true, y_pred, queues: list[str]) -> None:
    with tempfile.TemporaryDirectory() as d:
        cm = pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=queues), index=queues, columns=queues
        )
        cm.to_csv(Path(d) / "test_confusion_matrix.csv")
        report = classification_report(y_true, y_pred, labels=queues, zero_division=0)
        (Path(d) / "test_classification_report.txt").write_text(report)
        mlflow.log_artifacts(d, artifact_path="evaluation")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="complaint-routing")
    p.add_argument("--c", type=float, default=4.0)
    p.add_argument("--max-features", type=int, default=100_000)
    p.add_argument("--min-df", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--alias", default="champion")
    p.add_argument("--reproduces", default=None, help="run id this run reproduces")
    p.add_argument("--only-candidate", action="store_true")
    p.add_argument("--result-file", default=None, help="write the candidate run id + metrics here")
    args = p.parse_args()

    mlflow.set_experiment(args.experiment)
    queues = config.queues()
    parts = split(load_dataset())
    tr, va, te = parts["train"], parts["val"], parts["test"]
    tags = lineage_tags() | ({"reproduces": args.reproduces} if args.reproduces else {})
    common = {
        "train_months": "..".join(config.TRAIN_MONTHS), "val_months": "..".join(config.VAL_MONTHS),
        "test_months": "..".join(config.TEST_MONTHS), "n_train": len(tr), "n_val": len(va),
        "n_test": len(te), "feature_window_days": WINDOW_DAYS, "feature_lag_days": LAG_DAYS,
        "sample_per_month": config.SAMPLE_PER_MONTH,
    }  # fmt: skip
    print(json.dumps(common | tags, indent=1))

    if not args.only_candidate:
        majority = MajorityBaseline().fit(tr["queue"])
        with mlflow.start_run(run_name="baseline-majority", tags=tags | {"model_type": "rule"}):
            mlflow.log_params(common | {"rule": f"always {majority.queue_}"})
            for name, part in (("val", va), ("test", te)):
                mlflow.log_metrics(scores(part["queue"], majority.predict(len(part)), queues, name))
        keywords = KeywordBaseline(fallback=majority.queue_)
        with mlflow.start_run(run_name="baseline-keywords", tags=tags | {"model_type": "rule"}):
            mlflow.log_params(common | {"rule": "ordered keyword regex, fallback to majority"})
            for name, part in (("val", va), ("test", te)):
                mlflow.log_metrics(
                    scores(part["queue"], keywords.predict(part["narrative"]), queues, name)
                )

    params = {"model": "tfidf_1-2gram+onehot+company_mix -> logreg(balanced)",
              "C": args.c, "max_features": args.max_features, "min_df": args.min_df,
              "seed": args.seed}  # fmt: skip
    with mlflow.start_run(
        run_name="candidate-tfidf-logreg", tags=tags | {"model_type": "trained"}
    ) as run:
        mlflow.log_params(common | params)
        model = build_model(queues, args.c, args.max_features, args.min_df, args.seed)
        model.fit(tr, tr["queue"])
        metrics = {}
        for name, part in (("val", va), ("test", te)):
            metrics |= scores(part["queue"], model.predict(part), queues, name)
        mlflow.log_metrics(metrics)
        log_report(te["queue"], model.predict(te), queues)
        info = mlflow.sklearn.log_model(
            model, name="model", input_example=te.drop(columns="queue").head(3),
            registered_model_name=None if args.no_register else MODEL_NAME,
        )  # fmt: skip
        print(json.dumps({"run_id": run.info.run_id, **{k: round(v, 4) for k, v in metrics.items()
                          if "recall_" not in k}}, indent=1))  # fmt: skip

    if not args.no_register:
        client = MlflowClient()
        version = info.registered_model_version
        for key in ("data_version", "git_commit", "feature_version", "mapping_version"):
            client.set_model_version_tag(MODEL_NAME, version, key, tags[key])
        client.set_registered_model_alias(MODEL_NAME, args.alias, version)
        print(f"registered {MODEL_NAME} v{version}, alias '{args.alias}'")
    if args.result_file:
        Path(args.result_file).write_text(
            json.dumps({"run_id": run.info.run_id, "metrics": metrics})
        )


if __name__ == "__main__":
    main()
