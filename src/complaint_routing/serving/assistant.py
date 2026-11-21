"""The assistant service (the LLM service of Lab 4): POST /ask, a health contract, /metrics.

    uvicorn complaint_routing.serving.assistant:app --port 8000

A thin container around the Lab 3 query path. The LLM itself runs outside the cluster (vLLM on
the laptop, or an API); this service embeds the question, retrieves from pgvector and calls the
LLM over HTTP. Its whole configuration is config/llm.yaml (mounted from a ConfigMap, path in
$LLM_CONFIG) plus `KEY=VALUE` overrides in $LLM_SET for what differs inside the cluster (the
LLM's address); the effective configuration's hash is reported on every answer, as in Lab 3.

Ready = the embedding model is loaded, the index version for this config exists, and the LLM
server lists the configured model (checked at most every LLM_CHECK_S seconds).
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from complaint_routing.llm import settings
from complaint_routing.llm.index import Retriever, embed
from complaint_routing.llm.rag import Assistant

log = logging.getLogger("assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

IMAGE_COMMIT = os.environ.get("GIT_COMMIT", "unknown")
CONFIG = Path(os.environ.get("LLM_CONFIG", settings.CONFIG_PATH))
OVERRIDES = [s for s in os.environ.get("LLM_SET", "").split(";") if s.strip()]
RETRY_S = float(os.environ.get("LOAD_RETRY_S", "10"))
LLM_CHECK_S = float(os.environ.get("LLM_CHECK_S", "30"))

REQUESTS = Counter("assistant_requests_total", "Requests served", ["endpoint", "status"])
LATENCY = Histogram(
    "assistant_request_latency_seconds", "End-to-end /ask latency", ["endpoint"],
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60),
)  # fmt: skip
TOKENS = Counter("assistant_llm_tokens_total", "LLM tokens", ["direction"])
TOP_DISTANCE = Histogram(
    "assistant_top_distance", "Cosine distance of the best retrieved chunk",
    buckets=(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7),
)  # fmt: skip
READY = Gauge("assistant_ready", "1 when retrieval and the LLM server are usable")
INFO = Gauge(
    "assistant_info", "Configuration in use (value 1)",
    ["llm_model", "config_hash", "prompt_hash", "index_version", "image_commit"],
)  # fmt: skip


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Service:
    def __init__(self) -> None:
        self.cfg = settings.load(CONFIG, OVERRIDES)
        self.assistant: Assistant | None = None
        self.llm_ok_at = 0.0
        self.lock = threading.Lock()  # one retrieval connection, one LLM call at a time
        self.waiting: dict[str, str] = {"retrieval": "not loaded yet", "llm": "not checked yet"}
        self.thread = threading.Thread(target=self._load_until_ready, daemon=True, name="loader")

    @property
    def identity(self) -> dict:
        return {
            "llm_model": self.cfg["llm"]["model"],
            "config_hash": settings.config_hash(self.cfg),
            "prompt_hash": settings.prompt_hash(self.cfg, "answer"),
            "index_version": settings.index_version(self.cfg),
            "image_commit": IMAGE_COMMIT,
        }

    def _load_retrieval(self) -> None:
        embed(self.cfg["retrieval"]["embedding_model"], ["warm-up"])  # baked into the image
        self.assistant = Assistant(self.cfg, retriever=Retriever(self.cfg))

    def _check_llm(self) -> None:
        llm = self.assistant.llm if self.assistant else None
        if llm is None:
            from complaint_routing.llm.client import LLM

            llm = LLM.from_config(self.cfg)
        served = [m.id for m in llm._client.with_options(timeout=5).models.list()]
        if llm.model not in served:
            raise RuntimeError(f"{llm.base_url} serves {served}, not {llm.model}")
        self.llm_ok_at = time.monotonic()

    def _load_until_ready(self) -> None:
        INFO.labels(**self.identity).set(1)
        while "retrieval" in self.waiting:
            try:
                self._load_retrieval()
                self.waiting.pop("retrieval")
            except Exception as e:
                self.waiting["retrieval"] = f"{type(e).__name__}: {e}"[:300]
                log.warning("retrieval not ready: %s", self.waiting["retrieval"])
                time.sleep(RETRY_S)
        self.refresh_llm()

    def refresh_llm(self) -> None:
        """The LLM lives outside the cluster and can go away; re-check it on readiness probes."""
        if time.monotonic() - self.llm_ok_at < LLM_CHECK_S and "llm" not in self.waiting:
            return
        try:
            self._check_llm()
            self.waiting.pop("llm", None)
        except Exception as e:
            self.waiting["llm"] = f"{type(e).__name__}: {e}"[:300]
            log.warning("llm not ready: %s", self.waiting["llm"])
        READY.set(0 if self.waiting else 1)

    def ask(self, question: str) -> dict:
        with self.lock:
            a = self.assistant.answer(question)
        TOKENS.labels("in").inc(a.call.prompt_tokens)
        TOKENS.labels("out").inc(a.call.completion_tokens)
        if a.hits:
            TOP_DISTANCE.observe(a.hits[0].distance)
        return {
            "answer": a.call.text,
            "cited": a.cited,
            "retrieved": [{"doc_id": h.doc_id, "title": h.title, "url": h.url,
                           "distance": round(h.distance, 4)} for h in a.hits],  # fmt: skip
            "tokens_in": a.call.prompt_tokens,
            "tokens_out": a.call.completion_tokens,
            "llm_latency_s": round(a.call.latency_s, 3),
            **self.identity,
        }


service = Service()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    READY.set(0)
    service.thread.start()  # the server answers probes while the loader works
    yield


app = FastAPI(title="complaint assistant", version=IMAGE_COMMIT, lifespan=lifespan)




@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health/live")
def live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
@app.get("/health")
def ready():
    if "retrieval" not in service.waiting:
        service.refresh_llm()
    body = {"ready": not service.waiting, "waiting_for": service.waiting, **service.identity}
    return JSONResponse(body, status_code=200 if not service.waiting else 503)


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if service.waiting:
        REQUESTS.labels("/ask", "503").inc()
        raise HTTPException(503, detail={"not_ready": service.waiting})
    with LATENCY.labels("/ask").time():
        try:
            out = service.ask(req.question)
        except Exception:
            REQUESTS.labels("/ask", "500").inc()
            log.exception("answer failed")
            raise
    REQUESTS.labels("/ask", "200").inc()
    return out
