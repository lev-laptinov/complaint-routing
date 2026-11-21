"""The scheduled drift job (Kubernetes CronJob `drift-monitor`, every 5 minutes).

    python -m complaint_routing.monitoring.drift

1. Resolve the champion in the registry and download its reference window (reference.py).
2. Read the current window from the prediction log: the champion's predictions of the last
   DRIFT_WINDOW_MINUTES. Windows larger than DRIFT_MAX_ROWS are subsampled (seeded), so a
   p-value threshold means the same thing at any traffic level (design 6.6).
3. Evidently DataDriftPreset, every column with an explicit test: Kolmogorov-Smirnov for the
   numeric ones, chi-squared for the categorical ones, as Evidently does for small samples.
4. Prediction drift (the predicted queue is one of the columns, plus its Jensen-Shannon distance
   to the reference mix) and quality (macro-F1 on predictions whose labels arrived).
5. Push everything to the Prometheus Pushgateway; one MLflow run per window, with the Evidently
   report (HTML) attached when the alerting feature crosses its threshold.

The job also pushes when the window is too small (rows only): alerts are on the data, not on
the job's exit code (a green job with an empty window must still be visible).
"""

import logging
import os
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psycopg
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, f1_score

from complaint_routing import config
from complaint_routing.monitoring.reference import load_reference
from complaint_routing.registry import resolve
from complaint_routing.serving import predlog

log = logging.getLogger("drift")

MODEL_NAME = os.environ.get("MODEL_NAME", "complaint-router")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
WINDOW_MIN = int(os.environ.get("DRIFT_WINDOW_MINUTES", "10"))
MIN_ROWS = int(os.environ.get("DRIFT_MIN_ROWS", "100"))
MAX_ROWS = int(os.environ.get("DRIFT_MAX_ROWS", "300"))
QUALITY_LOOKBACK_H = int(os.environ.get("QUALITY_LOOKBACK_HOURS", "24"))
QUALITY_MIN_LABELS = int(os.environ.get("QUALITY_MIN_LABELS", "50"))
ALERT_FEATURE = os.environ.get("DRIFT_ALERT_FEATURE", "narrative_len")
ALERT_PVALUE = float(os.environ.get("DRIFT_ALERT_PVALUE", "0.001"))
PUSHGATEWAY = os.environ.get("PUSHGATEWAY_URL", "http://pushgateway:9091")
EXPERIMENT = os.environ.get("MONITORING_EXPERIMENT", "complaint-routing-monitoring")

TEXT = ["narrative_len", "n_words", "mask_share", "oov_share"]
CATEGORICAL = ["state", "submitted_via", "tags", "predicted_queue"]


def numeric_columns(queues: list[str]) -> list[str]:
    return TEXT + ["max_proba", "mix_n"] + [f"mix_{q}" for q in queues]


def read_window(conn, version: str, end: datetime) -> pd.DataFrame:
    cur = conn.execute(
        "SELECT * FROM serving.predictions WHERE model_version = %s AND ts > %s AND ts <= %s",
        (version, end - timedelta(minutes=WINDOW_MIN), end),
    )
    return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def read_labelled(conn, version: str, end: datetime) -> pd.DataFrame:
    cur = conn.execute(
        "SELECT p.predicted_queue, l.final_queue FROM serving.predictions p"
        " JOIN serving.labels l USING (complaint_id)"
        " WHERE p.model_version = %s AND p.ts > %s AND p.ts <= %s",
        (version, end - timedelta(hours=QUALITY_LOOKBACK_H), end),
    )
    return pd.DataFrame(cur.fetchall(), columns=["predicted_queue", "final_queue"])


def drift_scores(ref: pd.DataFrame, cur: pd.DataFrame, queues: list[str]):
    num, cat = numeric_columns(queues), CATEGORICAL
    dd = DataDefinition(numerical_columns=num, categorical_columns=cat)
    report = Report([DataDriftPreset(num_method="ks", cat_method="chisquare")])
    snap = report.run(
        Dataset.from_pandas(cur[num + cat].copy(), data_definition=dd),
        Dataset.from_pandas(ref[num + cat].copy(), data_definition=dd),
    )
    scores, share, drifted = {}, None, None
    for m in snap.dict()["metrics"]:
        c = m["config"]
        if "column" in c:
            scores[c["column"]] = (float(m["value"]), c.get("method", ""), c.get("threshold", 0.05))
        elif "drift_share" in c:
            share, drifted = float(m["value"]["share"]), float(m["value"]["count"])
    return snap, scores, share, drifted


def queue_mix_js(ref: pd.Series, cur: pd.Series, queues: list[str]) -> float:
    p = ref.value_counts(normalize=True).reindex(queues, fill_value=0)
    q = cur.value_counts(normalize=True).reindex(queues, fill_value=0)
    return float(jensenshannon(p, q, base=2))


def run(end: datetime | None = None, push: bool = True, log_mlflow: bool = True) -> dict:
    end = end or datetime.now(UTC)
    queues = config.queues()
    champion = resolve(MODEL_NAME, MODEL_ALIAS)
    reg = CollectorRegistry()

    def gauge(name: str, doc: str, value: float, **labels) -> None:
        g = gauge.made.get(name) or Gauge(name, doc, list(labels), registry=reg)
        gauge.made[name] = g
        (g.labels(**labels) if labels else g).set(value)

    gauge.made = {}
    gauge("drift_run_timestamp_seconds", "When the drift job last ran", end.timestamp())
    gauge("drift_window_minutes", "Length of the current window", WINDOW_MIN)
    gauge("drift_model_info", "Model version whose window and reference are compared", 1,
          model_version=champion.version)  # fmt: skip
    out: dict = {"model_version": champion.version, "window_end": end.isoformat()}

    with psycopg.connect(predlog.dsn(), connect_timeout=5) as conn:
        window = read_window(conn, champion.version, end)
        labelled = read_labelled(conn, champion.version, end)
        newest = conn.execute("SELECT extract(epoch FROM max(ts)) FROM serving.predictions")
        newest = newest.fetchone()[0]
    gauge("drift_window_rows", "Predictions of the champion in the current window", len(window))
    if newest is not None:
        gauge("prediction_log_newest_timestamp_seconds", "Newest row of the prediction log",
              float(newest))  # fmt: skip
    out["window_rows"] = len(window)

    if len(labelled) >= QUALITY_MIN_LABELS:
        y, p = labelled["final_queue"], labelled["predicted_queue"]
        out["quality_macro_f1"] = f1_score(y, p, labels=queues, average="macro", zero_division=0)
        out["quality_accuracy"] = accuracy_score(y, p)
        gauge("quality_macro_f1", "Macro-F1 on labelled predictions", out["quality_macro_f1"])
        gauge("quality_accuracy", "Accuracy on labelled predictions", out["quality_accuracy"])
    gauge("quality_labelled_rows", f"Labelled predictions, last {QUALITY_LOOKBACK_H} h",
          len(labelled))  # fmt: skip

    snap = None
    if len(window) >= MIN_ROWS:
        cur = window.sample(min(len(window), MAX_ROWS), random_state=int(end.timestamp()) // 60)
        ref = load_reference(champion.run_id)
        snap, scores, share, drifted = drift_scores(ref, cur, queues)
        for feature, (score, method, threshold) in scores.items():
            gauge("drift_feature_score", "Drift test result per column (p-value for KS / chi2)",
                  score, feature=feature, method=method)  # fmt: skip
            gauge("drift_feature_detected", "1 if Evidently's own threshold flags the column",
                  float(score < threshold), feature=feature)  # fmt: skip
        js = queue_mix_js(ref["predicted_queue"], cur["predicted_queue"], queues)
        gauge("drift_share", "Share of drifted columns (Evidently DriftedColumnsCount)", share)
        gauge("drift_columns_drifted", "Number of drifted columns", drifted)
        gauge("drift_window_rows_used", "Rows of the window the tests used", len(cur))
        gauge("prediction_queue_js_distance", "Jensen-Shannon distance, predicted queue mix",
              js)  # fmt: skip
        for q in queues:
            gauge("prediction_queue_share", "Share of predictions per queue",
                  float((cur["predicted_queue"] == q).mean()), queue=q, window="current")
            gauge("prediction_queue_share", "Share of predictions per queue",
                  float((ref["predicted_queue"] == q).mean()), queue=q, window="reference")
        alert_p = scores[ALERT_FEATURE][0]
        out |= {"rows_used": len(cur), "drift_share": share, "prediction_js": js,
                "alert_feature": ALERT_FEATURE, "alert_pvalue": alert_p,
                "alert": alert_p < ALERT_PVALUE,
                "scores": {k: round(v[0], 6) for k, v in scores.items()}}  # fmt: skip
    else:
        out["skipped"] = f"window has {len(window)} rows < {MIN_ROWS}"

    if push:
        push_to_gateway(PUSHGATEWAY, job="drift_monitor", registry=reg)
    if log_mlflow:
        log_run(out, snap)
    return out


def log_run(out: dict, snap) -> None:
    mlflow.set_experiment(EXPERIMENT)
    name = f"window-{out['window_end'][:16]}"
    tags = {"model_version": out["model_version"], "window_end": out["window_end"],
            "window_minutes": str(WINDOW_MIN), "alert_feature": ALERT_FEATURE,
            "alert_threshold": str(ALERT_PVALUE), "image_commit": os.environ.get("GIT_COMMIT", "")}
    with mlflow.start_run(run_name=name, tags=tags):  # fmt: skip
        mlflow.log_metrics({k: float(v) for k, v in out.items()
                            if isinstance(v, int | float) and not isinstance(v, bool)})  # fmt: skip
        if "scores" in out:
            mlflow.log_metrics({f"pvalue_{k}": v for k, v in out["scores"].items()})
            mlflow.log_dict(out, "drift_summary.json")
        if snap is not None and out.get("alert"):
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "evidently_drift_report.html"
                snap.save_html(str(path))
                mlflow.log_artifact(str(path))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = time.perf_counter()
    out = run()
    summary = {k: v for k, v in out.items() if k != "scores"}
    log.info("drift window: %s (%.1f s)", summary, time.perf_counter() - start)
    if "scores" in out:
        top = sorted(out["scores"].items(), key=lambda kv: kv[1])[:6]
        log.info("lowest p-values: %s", top)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
