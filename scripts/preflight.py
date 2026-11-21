"""Checks (and where cheap, creates) what the cluster's services need from Labs 2-3.

    uv run python scripts/preflight.py

- registry: complaint-router@champion exists (Lab 2 `make train`; fails with the command if not);
- reference window of that version (Lab 4; created here in ~20 s if missing);
- feature rows in features.complaint_features (Lab 2 ETL; fails with the command if empty);
- retrieval index for config/llm.yaml (Lab 3 `make rag-index`; built here in ~1 min if missing).
"""

import os
import sys

import psycopg
from mlflow.tracking import MlflowClient

from complaint_routing.monitoring.reference import ARTIFACT
from complaint_routing.registry import MODEL_NAME, resolve


def fail(msg: str) -> None:
    sys.exit(f"preflight: {msg}")


def main() -> None:
    client = MlflowClient()
    try:
        champion = resolve(MODEL_NAME, "champion", client)
    except Exception as e:
        fail(f"no {MODEL_NAME}@champion in {os.environ.get('MLFLOW_TRACKING_URI')} ({e});"
             " run Lab 2 first: make data-pull && make train")  # fmt: skip
    print(f"registry: {MODEL_NAME}@champion = v{champion.version} (run {champion.run_id[:8]})")

    files = [f.path for f in client.list_artifacts(champion.run_id, "monitoring")]
    if ARTIFACT in files:
        print(f"reference: {ARTIFACT} present")
    else:
        print("reference: missing, building it from the validation months")
        from complaint_routing.monitoring import reference

        sys.argv = ["reference"]
        reference.main()

    with psycopg.connect(os.environ["WAREHOUSE_DSN"], connect_timeout=5) as conn:
        try:
            n = conn.execute("SELECT count(*) FROM features.complaint_features").fetchone()[0]
        except psycopg.errors.UndefinedTable:
            n = 0
    if n == 0:
        fail("features.complaint_features is empty; run the Lab 2 ETL (make etl-backfill)")
    print(f"features: {n} rows in features.complaint_features")

    from complaint_routing.llm import index, settings

    cfg = settings.load()
    found = index.describe(cfg)
    if found is None:
        print(f"retrieval index {settings.index_version(cfg)}: missing, building it")
        print(index.build(cfg))
    else:
        print(f"retrieval index {found['index_version']}: {found['chunks']} chunks")


if __name__ == "__main__":
    main()
