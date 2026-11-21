"""Stretch goal: Alertmanager webhook -> a retraining Job, with nobody typing anything.

    uvicorn complaint_routing.monitoring.trigger:app --port 8080     # Deployment retrain-trigger

Alertmanager routes alerts labelled action="retrain" (InputDriftNarrativeLength) here. For each
firing alert the service creates a Job from the suspended CronJob `retrain` (the same thing
`kubectl create job --from=cronjob/retrain` does), annotated with the alert that started it,
unless a retraining Job is already running. It talks to the Kubernetes API with its own
ServiceAccount, which may only create and read Jobs and read CronJobs in this namespace.
"""

import json
import logging
import os
import time
from pathlib import Path

import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

log = logging.getLogger("trigger")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

SA = Path("/var/run/secrets/kubernetes.io/serviceaccount")
API = "https://kubernetes.default.svc"
CRONJOB = os.environ.get("RETRAIN_CRONJOB", "retrain")

RECEIVED = Counter("retrain_trigger_alerts_total", "Alerts received", ["alertname", "status"])
STARTED = Counter("retrain_trigger_jobs_started_total", "Retraining Jobs created", ["alertname"])
SKIPPED = Counter("retrain_trigger_skipped_total", "Alerts not acted on", ["reason"])

app = FastAPI(title="retrain trigger")


def kube(method: str, path: str, body: dict | None = None) -> dict:
    ns = (SA / "namespace").read_text().strip()
    r = requests.request(
        method,
        f"{API}{path.format(ns=ns)}",
        headers={"Authorization": f"Bearer {(SA / 'token').read_text().strip()}"},
        json=body,
        verify=str(SA / "ca.crt"),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def running_jobs() -> list[str]:
    jobs = kube("GET", "/apis/batch/v1/namespaces/{ns}/jobs?labelSelector=app%3Dretrain")
    return [j["metadata"]["name"] for j in jobs["items"] if j["status"].get("active")]


def start_job(alert: dict) -> str:
    cron = kube("GET", f"/apis/batch/v1/namespaces/{{ns}}/cronjobs/{CRONJOB}")
    spec = cron["spec"]["jobTemplate"]["spec"]
    labels = alert["labels"]
    name = f"retrain-{int(time.time())}"
    env = [{"name": "TRIGGER_ALERT", "value": labels.get("alertname", "")},
           {"name": "TRIGGER_STARTS_AT", "value": alert.get("startsAt", "")},
           {"name": "TRIGGER_FINGERPRINT", "value": alert.get("fingerprint", "")}]  # fmt: skip
    for c in spec["template"]["spec"]["containers"]:
        c["env"] = c.get("env", []) + env
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "labels": {"app": "retrain", "triggered-by": "alertmanager"},
            "annotations": {"complaint-routing/alert": labels.get("alertname", ""),
                            "complaint-routing/alert-starts-at": alert.get("startsAt", ""),
                            "complaint-routing/alert-summary":
                                alert.get("annotations", {}).get("summary", "")[:200]},
        },  # fmt: skip
        "spec": spec,
    }
    kube("POST", "/apis/batch/v1/namespaces/{ns}/jobs", job)
    return name


@app.post("/alert")
async def alert(request: Request) -> dict:
    payload = await request.json()
    started = []
    for a in payload.get("alerts", []):
        name = a.get("labels", {}).get("alertname", "?")
        RECEIVED.labels(name, a.get("status", "?")).inc()
        if a.get("status") != "firing" or a.get("labels", {}).get("action") != "retrain":
            SKIPPED.labels("not a firing retrain alert").inc()
            continue
        busy = running_jobs()
        if busy:
            SKIPPED.labels("retraining already running").inc()
            log.info("alert %s: retraining already running (%s)", name, busy)
            continue
        job = start_job(a)
        STARTED.labels(name).inc()
        started.append(job)
        log.info("alert %s (since %s) -> started Job %s", name, a.get("startsAt"), job)
    log.info("webhook: %s", json.dumps({"alerts": len(payload.get("alerts", [])),
                                        "started": started}))  # fmt: skip
    return {"started": started}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
