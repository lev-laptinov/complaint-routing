"""Monthly ETL for one batch of CFPB complaints.

extract -> validate -> transform -> load -> build_features -> quality_check

The batch month comes from the run's data interval (data_interval_start), or from the `month`
param of a manual run. Never from the wall clock: a backfill of 2023-04 processes 2023-04.
"""

import json
from pathlib import Path

import pandas as pd
import pendulum
from airflow.sdk import Param, dag, get_current_context, task
from airflow.timetables.interval import CronDataIntervalTimetable

STAGING = Path("/opt/airflow/staging")


def _batch_month() -> str:
    ctx = get_current_context()
    if ctx["params"].get("month"):
        return ctx["params"]["month"]
    if ctx["data_interval_start"] is None:  # Airflow 3 manual runs have no interval
        raise ValueError(
            "manual run without a month: pass conf {'month': 'YYYY-MM'} or use backfill"
        )
    return ctx["data_interval_start"].strftime("%Y-%m")


def _staging_dir(month: str) -> Path:
    ctx = get_current_context()
    d = STAGING / month / ctx["run_id"].replace(":", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


@dag(
    dag_id="complaints_etl",
    # Airflow 3: plain "@monthly" no longer creates intervals (start == end). Declare it explicitly.
    schedule=CronDataIntervalTimetable("@monthly", timezone="UTC"),
    start_date=pendulum.datetime(2022, 1, 1, tz="UTC"),
    end_date=pendulum.datetime(2026, 8, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,  # features of month M read months M-5..M-1, so months load in order
    default_args={"retries": 0},
    params={
        "month": Param(
            None, type=["null", "string"], description="YYYY-MM; overrides the interval"
        ),
        "raw_path": Param(None, type=["null", "string"], description="file under data/, for tests"),
    },
    tags=["lab-2"],
)
def complaints_etl():
    @task
    def extract() -> dict:
        from complaint_routing import config, lineage

        month = _batch_month()
        override = get_current_context()["params"].get("raw_path")
        src = config.ROOT / override if override else config.raw_path(month)
        if not src.exists():
            raise FileNotFoundError(f"no raw batch for {month}: {src}")
        raw = pd.read_parquet(src)
        out = _staging_dir(month) / "raw.parquet"
        raw.to_parquet(out, index=False)
        md5 = lineage.file_md5(src)
        version = f"unversioned:{md5}" if override else lineage.data_version()
        info = {"month": month, "source": str(src), "source_md5": md5, "data_version": version,
                "raw_path": str(out), "raw_rows": len(raw)}  # fmt: skip
        print(json.dumps(info))
        return info

    @task
    def validate(info: dict) -> dict:
        from complaint_routing import config, lineage, quality

        raw = pd.read_parquet(info["raw_path"])
        failures = quality.validate_raw(raw, info["month"], config.load_mapping())
        # Stale partition: the same file delivered again for a different month.
        prev = config.months_between("2000-01", info["month"])[-2]
        if (
            config.raw_path(prev).exists()
            and lineage.file_md5(config.raw_path(prev)) == info["source_md5"]
        ):
            failures.append(f"stale batch: identical to {prev}")
        if failures:
            raise ValueError(
                "validate failed for " + info["month"] + ":\n- " + "\n- ".join(failures)
            )
        print(f"validate passed: {len(raw)} rows")
        return info

    @task
    def transform(info: dict) -> dict:
        from complaint_routing import config
        from complaint_routing.transform import clean

        df, stats = clean(pd.read_parquet(info["raw_path"]), config.load_mapping())
        df["data_version"] = info["data_version"]
        out = Path(info["raw_path"]).with_name("clean.parquet")
        df.to_parquet(out, index=False)
        print(json.dumps(stats))
        return {**info, "clean_path": str(out), "clean_rows": len(df), "clean_stats": stats}

    @task
    def load(info: dict) -> dict:
        from complaint_routing import warehouse

        eng = warehouse.engine()
        month = info["month"]
        raw = pd.read_parquet(info["raw_path"])
        raw.columns = [
            c.lower().replace(" ", "_").replace("?", "").replace("-", "_") for c in raw.columns
        ]
        raw = raw.assign(batch_month=month, data_version=info["data_version"])
        clean = pd.read_parquet(info["clean_path"])
        counts = {
            "raw": warehouse.replace_month(eng, warehouse.TABLES["raw"], month, raw),
            "dwh": warehouse.replace_month(eng, warehouse.TABLES["dwh"], month, clean),
        }
        print(f"rows for {month} after load: {counts}")
        return {**info, "loaded": counts}

    @task
    def build_features(info: dict) -> dict:
        from complaint_routing import config, warehouse
        from complaint_routing.features import company_queue_mix

        eng = warehouse.engine()
        month = info["month"]
        window = config.months_between("2000-01", month)[
            -6:
        ]  # M-5..M: covers lag 60 + window 90 days
        hist = warehouse.read_sql(
            eng,
            "SELECT complaint_id, company, date_received, queue, batch_month "
            "FROM dwh.complaints WHERE batch_month = ANY(:months)",
            months=window,
        )
        hist["date_received"] = pd.to_datetime(hist["date_received"])
        targets = hist[hist["batch_month"] == month][["complaint_id", "company", "date_received"]]
        feats = company_queue_mix(hist, targets, config.queues())
        feats = feats.rename(columns={"date_received": "event_timestamp"}).assign(batch_month=month)
        n = warehouse.replace_month(eng, warehouse.TABLES["features"], month, feats)
        history_months = sorted(set(hist["batch_month"]) - {month})
        print(f"features for {month}: {n} rows; history months present: {history_months}")
        return {**info, "feature_rows": n}

    @task
    def quality_check(info: dict) -> None:
        from complaint_routing import quality, warehouse
        from complaint_routing.features import FEATURE_VERSION

        eng = warehouse.engine()
        m = info["month"]
        one = lambda q: warehouse.read_sql(eng, q, m=m).iloc[0, 0]  # noqa: E731
        counts = {
            "dwh": one("SELECT count(*) FROM dwh.complaints WHERE batch_month = :m"),
            "features": one(
                "SELECT count(*) FROM features.complaint_features WHERE batch_month = :m"
            ),
            "outside_month": one(
                "SELECT count(*) FROM dwh.complaints WHERE batch_month = :m "
                "AND to_char(date_received, 'YYYY-MM') <> :m"
            ),
        }
        null_queue = one(
            "SELECT count(*) FROM dwh.complaints WHERE batch_month = :m AND queue IS NULL"
        )
        versions = warehouse.read_sql(
            eng,
            "SELECT DISTINCT feature_version FROM features.complaint_features "
            "WHERE batch_month = :m",
            m=m,
        )["feature_version"].tolist()
        failures = quality.check_loaded(
            m, info["clean_rows"], counts, FEATURE_VERSION, versions, null_queue
        )
        if failures:
            raise ValueError("quality_check failed for " + m + ":\n- " + "\n- ".join(failures))
        print(f"quality_check passed for {m}: {counts}")

    quality_check(build_features(load(transform(validate(extract())))))


complaints_etl()
