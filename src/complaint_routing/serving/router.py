"""The routing service (the ML service of Lab 4): POST /predict, a health contract, /metrics.

    uvicorn complaint_routing.serving.router:app --port 8000

Start-up never blocks the server: a background thread keeps trying to
1. resolve models:/$MODEL_NAME@$MODEL_ALIAS in the registry and load that exact version,
2. load the point-in-time feature rows the model's feature_version needs,
3. open the prediction log,
and /health/ready answers 503 with what it is still waiting for until all three are done.
/health/live only says the process is alive: restarting a pod cannot create a missing model,
so a missing model keeps the pod out of the Service (not ready) instead of in a restart loop.
"""

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from complaint_routing import config
from complaint_routing.registry import Resolved, resolve
from complaint_routing.serving import predlog
from complaint_routing.serving.features import (
    MixLookup,
    describe_text,
    load_mix_lookup,
    text_analyzer,
    text_vocabulary,
)
from complaint_routing.transform import scrub_pii

log = logging.getLogger("router")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

MODEL_NAME = os.environ.get("MODEL_NAME", "complaint-router")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
IMAGE_COMMIT = os.environ.get("GIT_COMMIT", "unknown")
RETRY_S = float(os.environ.get("LOAD_RETRY_S", "10"))

REQUESTS = Counter("router_requests_total", "Requests served", ["endpoint", "status"])
LATENCY = Histogram(
    "router_request_latency_seconds", "End-to-end /predict latency", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5),
)  # fmt: skip
PREDICTIONS = Counter(
    "router_predictions_total", "Predictions by queue", ["queue", "model_version"]
)
CONFIDENCE = Histogram(
    "router_prediction_max_proba", "Highest class probability per prediction",
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)  # fmt: skip
LOG_ERRORS = Counter("router_prediction_log_errors_total", "Predictions that were not logged")
READY = Gauge("router_ready", "1 when the model, features and prediction log are loaded")
MODEL_INFO = Gauge(
    "router_model_info", "The loaded model version (value 1)",
    ["model_name", "model_alias", "model_version", "image_commit"],
)  # fmt: skip


class PredictRequest(BaseModel):
    narrative: str = Field(min_length=1)
    company: str = "none"
    state: str = "none"
    submitted_via: str = "Web"
    tags: str = "none"
    complaint_id: int | None = None
    # Live intake leaves it empty (= now). The replay passes the complaint's original date.
    date_received: datetime | None = None


class Service:
    def __init__(self) -> None:
        self.queues = config.queues()
        self.resolved: Resolved | None = None
        self.model = None
        self.mix: MixLookup | None = None
        self.db: psycopg.Connection | None = None
        self.db_lock = threading.Lock()
        self.waiting: dict[str, str] = {
            "model": "not loaded yet",
            "features": "not loaded yet",
            "prediction_log": "not open yet",
        }
        self.thread = threading.Thread(target=self._load_until_ready, daemon=True, name="loader")

    # --- start-up -------------------------------------------------------------------------
    def _load_model(self) -> None:
        r = resolve(MODEL_NAME, MODEL_ALIAS)
        expected = config.mapping_version()
        if r.tags.get("mapping_version") not in (None, expected):
            raise RuntimeError(
                f"v{r.version} was trained on queue mapping {r.tags['mapping_version']},"
                f" this image has {expected}: the queue lists differ"
            )
        t = time.perf_counter()
        model = mlflow.sklearn.load_model(r.uri)
        self.vocab, self.analyze = text_vocabulary(model), text_analyzer(model)
        self.model, self.resolved = model, r
        MODEL_INFO.labels(MODEL_NAME, MODEL_ALIAS, r.version, IMAGE_COMMIT).set(1)
        log.info("loaded %s (alias %s) in %.1f s", r.uri, MODEL_ALIAS, time.perf_counter() - t)

    def _open_db(self) -> None:
        conn = psycopg.connect(predlog.dsn(), connect_timeout=5)
        predlog.ensure_schema(conn, self.queues)
        self.db = conn

    def _load_features(self) -> None:
        fv = self.resolved.tags.get("feature_version", "v1")
        self.mix = load_mix_lookup(self.db, self.queues, fv)
        if len(self.mix) == 0:
            self.mix = None
            raise RuntimeError(f"features.complaint_features has no rows of feature_version {fv}")
        log.info("feature rows loaded for %d companies (feature_version %s)", len(self.mix), fv)

    def _load_until_ready(self) -> None:
        steps = [("model", self._load_model, lambda: self.model is not None),
                 ("prediction_log", self._open_db, lambda: self.db is not None),
                 ("features", self._load_features, lambda: self.mix is not None)]  # fmt: skip
        while self.waiting:
            for name, step, done in steps:
                if done() or (name == "features" and (self.model is None or self.db is None)):
                    continue
                try:
                    step()
                    self.waiting.pop(name, None)
                except Exception as e:  # registry, alias, artifact store or database not there
                    self.waiting[name] = f"{type(e).__name__}: {e}"[:300]
                    log.warning("%s not ready: %s", name, self.waiting[name])
            READY.set(0 if self.waiting else 1)
            if self.waiting:
                time.sleep(RETRY_S)

    # --- serving --------------------------------------------------------------------------
    def predict(self, req: PredictRequest) -> dict:
        start = time.perf_counter()
        narrative = scrub_pii(req.narrative.strip())
        at = (req.date_received or datetime.now(UTC)).replace(tzinfo=None)
        fields = {"narrative": narrative, "company": req.company, "state": req.state,
                  "submitted_via": req.submitted_via, "tags": req.tags}  # fmt: skip
        mix = self.mix.get(req.company, pd.Timestamp(at))
        x = pd.DataFrame([fields | mix])
        proba = self.model.predict_proba(x)[0]
        classes = [str(c) for c in self.model.classes_]
        best = int(np.argmax(proba))
        queue = classes[best]
        latency = time.perf_counter() - start
        request_id = uuid.uuid4()
        text = describe_text(pd.Series([narrative]), self.vocab, self.analyze).iloc[0].to_dict()
        self._log(
            {"request_id": request_id, "model_name": MODEL_NAME,
             "model_version": self.resolved.version, "image_commit": IMAGE_COMMIT,
             "complaint_id": req.complaint_id,
             "date_received": req.date_received.replace(tzinfo=None) if req.date_received else None,
             **fields, **text, **mix, "predicted_queue": queue, "max_proba": float(proba[best]),
             "probabilities": dict(zip(classes, proba.round(4).tolist(), strict=True)),
             "latency_ms": latency * 1000}  # fmt: skip
        )
        PREDICTIONS.labels(queue, self.resolved.version).inc()
        CONFIDENCE.observe(float(proba[best]))
        return {
            "request_id": str(request_id),
            "queue": queue,
            "probabilities": dict(zip(classes, proba.round(4).tolist(), strict=True)),
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "model_version": self.resolved.version,
            "image_commit": IMAGE_COMMIT,
        }

    def _log(self, row: dict) -> None:
        """A prediction that is not logged is invisible to monitoring: count it, retry once."""
        with self.db_lock:
            for attempt in (1, 2):
                try:
                    predlog.insert(self.db, row)
                    return
                except Exception as e:
                    log.warning("prediction log insert failed (attempt %d): %s", attempt, e)
                    try:
                        self.db.close()
                        self._open_db()
                    except Exception as e2:
                        log.warning("prediction log reconnect failed: %s", e2)
        LOG_ERRORS.inc()


service = Service()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    READY.set(0)
    service.thread.start()  # the server answers probes while the loader works
    yield


app = FastAPI(title="complaint router", version=IMAGE_COMMIT, lifespan=lifespan)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)




@app.get("/health/live")
def live() -> dict:
    return {"status": "alive", "loader_running": service.thread.is_alive() or not service.waiting}


@app.get("/health/ready")
@app.get("/health")
def ready():
    body = {
        "ready": not service.waiting,
        "waiting_for": service.waiting,
        "model": f"{MODEL_NAME}@{MODEL_ALIAS}",
        "model_version": service.resolved.version if service.resolved else None,
        "image_commit": IMAGE_COMMIT,
    }
    return JSONResponse(body, status_code=200 if not service.waiting else 503)


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if service.waiting:
        REQUESTS.labels("/predict", "503").inc()
        raise HTTPException(503, detail={"not_ready": service.waiting})
    with LATENCY.labels("/predict").time():
        try:
            out = service.predict(req)
        except Exception:
            REQUESTS.labels("/predict", "500").inc()
            log.exception("prediction failed")
            raise
    REQUESTS.labels("/predict", "200").inc()
    return out
