import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from complaint_routing import config
from complaint_routing.features import mix_columns
from complaint_routing.monitoring import shift
from complaint_routing.monitoring.drift import queue_mix_js
from complaint_routing.registry import Resolved
from complaint_routing.serving import router
from complaint_routing.serving.features import MixLookup, describe_text
from complaint_routing.train import build_model

QUEUES = config.queues()


def test_mix_lookup_is_point_in_time():
    cols = mix_columns(QUEUES)
    rows = pd.DataFrame(
        [{"company": "A", "event_timestamp": pd.Timestamp(d), **dict.fromkeys(cols, v)}
         for d, v in [("2026-03-01", 0.1), ("2026-03-10", 0.2)]]  # fmt: skip
    )
    mix = MixLookup.from_frame(rows, QUEUES)
    assert mix.get("A", pd.Timestamp("2026-02-28"))["mix_card"] == 0.0  # before the first row
    assert mix.get("A", pd.Timestamp("2026-03-05"))["mix_card"] == 0.1  # the earlier row
    assert mix.get("A", pd.Timestamp("2026-03-10"))["mix_card"] == 0.2  # same day counts
    unknown = mix.get("B", pd.Timestamp("2026-03-05"))
    assert unknown["mix_n"] == 0 and set(unknown) == set(cols)


def test_text_descriptors_and_oov():
    text = pd.Series(["the loan XXXX was late", "zzqx"])
    out = describe_text(text, vocab={"the", "loan", "was", "late"}, analyze=str.split)
    assert out["n_words"].tolist() == [5, 1]
    assert out["mask_share"].tolist() == [0.2, 0.0]
    assert out["oov_share"].tolist() == [0.2, 1.0]  # XXXX and zzqx are not in the vocabulary


def test_truncate_shift_touches_the_documented_fraction():
    df = pd.DataFrame({"narrative": [" ".join(["w"] * 100)] * 1000})
    out, hit = shift.truncate(df, fraction=0.3, words=25, seed=7)
    assert 0.25 < hit.mean() < 0.35
    words = out["narrative"].str.split().str.len()
    assert (words[hit] == 25).all() and (words[~hit] == 100).all()
    assert shift.apply(df, "none", 0.3, 25, 7)[1].sum() == 0


def test_queue_mix_distance():
    a = pd.Series(["card"] * 50 + ["mortgage"] * 50)
    assert queue_mix_js(a, a, QUEUES) == pytest.approx(0.0)
    assert queue_mix_js(a, pd.Series(["debt_collection"] * 10), QUEUES) == pytest.approx(1.0)


@pytest.fixture
def client(monkeypatch):
    """The real app without its loader thread: TestClient without `with` skips start-up."""
    svc = router.Service()
    monkeypatch.setattr(router, "service", svc)
    return TestClient(router.app), svc


def test_not_ready_until_loaded_and_predict_refused(client):
    c, _ = client
    r = c.get("/health/ready")
    assert r.status_code == 503 and set(r.json()["waiting_for"]) == {
        "model", "features", "prediction_log"}  # fmt: skip
    assert c.get("/health/live").status_code == 200
    assert c.post("/predict", json={"narrative": "my mortgage"}).status_code == 503


def test_predict_once_loaded(client):
    c, svc = client
    rng = np.random.default_rng(0)
    words = {"card": "credit card charge", "mortgage": "mortgage escrow home"}
    rows = []
    for i in range(80):
        q = list(words)[i % 2]
        rows.append({"narrative": f"{words[q]} {rng.integers(1000)}", "company": "A",
                     "state": "TX", "submitted_via": "Web", "tags": "none", "queue": q,
                     **dict.fromkeys(mix_columns(QUEUES), 0.0)})  # fmt: skip
    train = pd.DataFrame(rows)
    model = build_model(QUEUES, c=1.0, max_features=100, min_df=1, seed=0)
    model.fit(train, train["queue"])
    svc.model, svc.resolved = model, Resolved("complaint-router", "champion", "7", "run", {})
    svc.vocab, svc.analyze = {"mortgage"}, str.split
    svc.mix = MixLookup(QUEUES)
    logged = []
    svc._log = logged.append
    svc.waiting = {}
    r = c.post("/predict", json={"narrative": "escrow on my mortgage, call 555-123-4567"})
    assert r.status_code == 200
    body = r.json()
    assert body["queue"] == "mortgage" and body["model_version"] == "7"
    assert c.get("/health/ready").status_code == 200
    assert "XXXX" in logged[0]["narrative"] and "555" not in logged[0]["narrative"]  # scrubbed
    assert logged[0]["predicted_queue"] == "mortgage" and logged[0]["model_version"] == "7"
