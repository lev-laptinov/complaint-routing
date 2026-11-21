"""Stretch goal: the retraining run an alert starts (Job from the suspended CronJob `retrain`).

    python -m complaint_routing.monitoring.retrain

Trigger-based retraining on a sliding window of the most recent labelled data:
- training rows: the last RETRAIN_MONTHS months before the validation window, from the
  warehouse (dwh.complaints + features.complaint_features, both written by the Lab 2 ETL),
  plus every prediction-log row whose label has arrived (serving.labels), so the window the
  services saw is in the training set as soon as it is labelled;
- the champion's own hyperparameters (read from its MLflow run);
- evaluation on the validation months, the same rows the champion was compared on.

The result is registered as a new version with alias `candidate`, never `champion`: promotion
stays a human decision (design 5.5). The run carries what started it (alert name and time),
the image commit, the data window and the comparison with the champion.
"""

import logging
import os

import mlflow
import pandas as pd
import psycopg
from mlflow.tracking import MlflowClient

from complaint_routing import config
from complaint_routing.features import FEATURE_VERSION, mix_columns
from complaint_routing.monitoring.reference import log_reference
from complaint_routing.registry import MODEL_NAME, resolve
from complaint_routing.serving import predlog
from complaint_routing.train import build_model, scores

log = logging.getLogger("retrain")

MONTHS = int(os.environ.get("RETRAIN_MONTHS", "12"))
ALIAS = os.environ.get("RETRAIN_ALIAS", "candidate")
EXPERIMENT = os.environ.get("RETRAIN_EXPERIMENT", "complaint-routing")
COLUMNS = ["narrative", "company", "state", "submitted_via", "tags", "queue", "batch_month"]


def warehouse_rows(conn, months: list[str], queues: list[str]) -> pd.DataFrame:
    mix = ", ".join(f"f.{c}" for c in mix_columns(queues))
    cur = conn.execute(
        "SELECT d.complaint_id, d.date_received, d.narrative, d.company, d.state,"
        f" d.submitted_via, d.tags, d.queue, d.batch_month, {mix}"
        " FROM dwh.complaints d JOIN features.complaint_features f USING (complaint_id)"
        " WHERE d.batch_month = ANY(%s) AND f.feature_version = %s",
        (months, FEATURE_VERSION),
    )
    return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def labelled_serving_rows(conn, queues: list[str]) -> pd.DataFrame:
    cols = ", ".join(f"p.{c}" for c in mix_columns(queues))
    cur = conn.execute(
        "SELECT p.complaint_id, p.date_received, p.narrative, p.company, p.state,"
        f" p.submitted_via, p.tags, l.final_queue AS queue, {cols}"
        " FROM serving.predictions p JOIN serving.labels l USING (complaint_id)"
    )
    df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    return df.drop_duplicates("complaint_id").assign(batch_month="serving")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    queues = config.queues()
    champion = resolve(MODEL_NAME, "champion")
    client = MlflowClient()
    params = client.get_run(champion.run_id).data.params
    champ_metrics = client.get_run(champion.run_id).data.metrics

    last_train = config.months_between("2000-01", config.VAL_MONTHS[0])[-2]
    train_months = config.months_between("2000-01", last_train)[-MONTHS:]
    val_months = config.months_between(*config.VAL_MONTHS)
    with psycopg.connect(predlog.dsn(), connect_timeout=5) as conn:
        hist = warehouse_rows(conn, train_months, queues)
        val = warehouse_rows(conn, val_months, queues)
        served = labelled_serving_rows(conn, queues)
    train = pd.concat([hist, served], ignore_index=True)
    log.info("training rows: %d from warehouse %s..%s + %d labelled serving rows; val %d",
             len(hist), train_months[0], train_months[-1], len(served), len(val))  # fmt: skip

    tags = {
        "trigger": f"alert:{os.environ.get('TRIGGER_ALERT', 'manual')}",
        "trigger_starts_at": os.environ.get("TRIGGER_STARTS_AT", ""),
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "data_version": champion.tags.get("data_version", ""),
        "feature_version": FEATURE_VERSION,
        "mapping_version": config.mapping_version(),
        "retrained_from": f"v{champion.version}",
        "model_type": "trained",
    }
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="retrain-on-alert", tags=tags) as run:
        hp = {"c": float(params.get("C", 4.0)),
              "max_features": int(params.get("max_features", 100000)),
              "min_df": int(params.get("min_df", 3)),
              "seed": int(params.get("seed", 42))}  # fmt: skip
        mlflow.log_params(hp | {"train_months": f"{train_months[0]}..{train_months[-1]}",
                                "n_train": len(train), "n_serving_labelled": len(served),
                                "n_val": len(val)})  # fmt: skip
        model = build_model(queues, **hp)
        model.fit(train, train["queue"])
        metrics = scores(val["queue"], model.predict(val), queues, "val")
        mlflow.log_metrics(metrics)
        log_reference(model, val, queues)
        champ_f1 = champ_metrics.get("val_macro_f1", float("nan"))
        gate = metrics["val_macro_f1"] >= champ_f1 and metrics["val_min_recall"] >= 0.60
        mlflow.set_tags({"champion_val_macro_f1": f"{champ_f1:.4f}",
                         "promotion_gate": "pass" if gate else "fail"})  # fmt: skip
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME)
    version = info.registered_model_version
    for k, v in tags.items():
        client.set_model_version_tag(MODEL_NAME, version, k, v)
    client.set_model_version_tag(MODEL_NAME, version, "promotion_gate", "pass" if gate else "fail")
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    log.info(
        "registered %s v%s as @%s (run %s): val macro-F1 %.4f vs champion v%s %.4f, min recall"
        " %.3f -> promotion gate %s; promotion stays manual: make promote VERSION=%s",
        MODEL_NAME, version, ALIAS, run.info.run_id, metrics["val_macro_f1"], champion.version,
        champ_f1, metrics["val_min_recall"], "PASS" if gate else "FAIL", version,
    )  # fmt: skip


if __name__ == "__main__":
    main()
