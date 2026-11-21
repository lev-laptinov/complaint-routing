"""Replay held-out complaints through the routing service, unchanged or with a documented shift.

    uv run python -m complaint_routing.monitoring.replay --rows 300                    # baseline
    uv run python -m complaint_routing.monitoring.replay --rows 300 --shift truncate   # shifted

Rows: a seeded sample of the test months (2026-05 -> 07), never seen in training or in the
reference window, cleaned with the pipeline's clean() from the DVC-versioned raw files. Each row
is sent to POST /predict as the intake service would send it (narrative and form fields, plus
its original date so the service looks up the features of that day). The service logs every
request; that log, not this script, is what the drift job reads.

--labels also delivers the true queues to serving.labels (the case-closure feed), as if the
complaints had closed; the drift job then reports macro-F1 on them.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import psycopg
import requests

from complaint_routing import config
from complaint_routing.monitoring import shift
from complaint_routing.serving import predlog
from complaint_routing.transform import clean

FIELDS = ["narrative", "company", "state", "submitted_via", "tags"]


def held_out(rows: int, seed: int) -> pd.DataFrame:
    mapping = config.load_mapping()
    months = config.months_between(*config.TEST_MONTHS)
    df = pd.concat([clean(pd.read_parquet(config.raw_path(m)), mapping)[0] for m in months])
    return df.sample(rows, random_state=seed).reset_index(drop=True)


def send(url: str, row: dict, timeout: float) -> dict:
    body = {k: row[k] for k in FIELDS} | {
        "complaint_id": int(row["complaint_id"]),
        "date_received": pd.Timestamp(row["date_received"]).isoformat(),
    }
    t = time.perf_counter()
    r = requests.post(f"{url}/predict", json=body, timeout=timeout)
    out = {"status": r.status_code, "latency_s": time.perf_counter() - t}
    if r.ok:
        out |= {k: r.json()[k] for k in ("queue", "model_version")}
    return out


def deliver_labels(df: pd.DataFrame) -> int:
    with psycopg.connect(predlog.dsn()) as conn:
        predlog.ensure_schema(conn, config.queues())
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO serving.labels (complaint_id, final_queue) VALUES (%s, %s)"
                " ON CONFLICT (complaint_id) DO UPDATE SET final_queue = EXCLUDED.final_queue,"
                " closed_at = now()",
                [(int(i), q) for i, q in zip(df["complaint_id"], df["queue"], strict=True)],
            )
    return len(df)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8081")
    p.add_argument("--rows", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--shift", choices=shift.SHIFTS, default="none")
    p.add_argument("--fraction", type=float, default=0.3, help="share of rows the shift touches")
    p.add_argument("--words", type=int, default=25, help="truncate: words kept")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--labels", action="store_true", help="also deliver the true queues")
    args = p.parse_args()

    df = held_out(args.rows, args.seed)
    df, hit = shift.apply(df, args.shift, args.fraction, args.words, args.seed)
    words = df["narrative"].str.split().str.len()
    print(
        f"{len(df)} held-out complaints (test months {'..'.join(config.TEST_MONTHS)}, seed"
        f" {args.seed}); shift={args.shift}: {int(hit.sum())} rows changed"
        f" ({hit.mean():.0%}); median words {int(words.median())}, median chars"
        f" {int(df['narrative'].str.len().median())}"
    )
    start = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(
            pool.map(lambda r: send(args.url, r, args.timeout), df.to_dict("records"))
        )
    res = pd.DataFrame(results)
    took = time.perf_counter() - start
    ok = res["status"].eq(200)
    summary = {
        "sent": len(res),
        "ok": int(ok.sum()),
        "errors": res.loc[~ok, "status"].value_counts().to_dict(),
        "seconds": round(took, 1),
        "latency_p50_ms": round(res["latency_s"].quantile(0.5) * 1000, 1),
        "latency_p95_ms": round(res["latency_s"].quantile(0.95) * 1000, 1),
        "model_versions": res.get("model_version", pd.Series(dtype=str)).value_counts().to_dict(),
    }
    if ok.any():
        summary["accuracy_vs_true_queue"] = round(
            float((res.loc[ok, "queue"] == df.loc[ok, "queue"]).mean()), 3
        )
    print(json.dumps(summary))
    if args.labels:
        print(f"labels delivered to serving.labels: {deliver_labels(df[ok])}")


if __name__ == "__main__":
    main()
