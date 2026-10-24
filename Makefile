-include .env
export

MONTH ?= 2026-05
AIRFLOW = docker compose exec -T airflow airflow
PSQL = docker compose exec -T postgres psql -U $(POSTGRES_USER)

setup:  ## pinned environment, pre-commit hook, local .env
	[ -f .env ] || cp .env.example .env
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .

test:
	uv run pytest -q

up:  ## PostgreSQL, MinIO, MLflow, Airflow
	docker compose up -d --build
	@echo "Airflow http://localhost:8080  MLflow http://localhost:5050  MinIO http://localhost:9311"

down:
	docker compose down

## --- data (DVC, remote = MinIO bucket s3://dvc) ---
data-pull:
	uv run dvc pull

data-push:
	uv run dvc push

data-rebuild:  ## re-download the public CFPB archive and rebuild data/raw byte for byte
	uv run python -m complaint_routing.ingest --verify

## --- pipeline (Airflow DAG complaints_etl) ---
etl-backfill:  ## every month 2022-01..2026-07, in order
	$(AIRFLOW) backfill create --dag-id complaints_etl --from-date 2022-01-01 --to-date 2026-07-01 --max-active-runs 1

etl-month:  ## one interval, rerun even if it already succeeded: make etl-month MONTH=2026-05
	$(AIRFLOW) backfill create --dag-id complaints_etl --from-date $(MONTH)-01 --to-date $(MONTH)-01 --reprocess-behavior completed

etl-wait:
	@until [ "$$($(PSQL) -d airflow -At -c "select (select count(*) from dag_run where state in ('queued','running')) + (select count(*) from backfill where completed_at is null)")" = "0" ]; do sleep 5; done
	@$(PSQL) -d airflow -c "select run_id, state, data_interval_start, end_date from dag_run order by end_date desc nulls first limit 3"

counts:  ## rows of one batch month in each layer
	@$(PSQL) -d warehouse -c "select 'raw' as layer, count(*) from raw.complaints where batch_month = '$(MONTH)' union all select 'dwh', count(*) from dwh.complaints where batch_month = '$(MONTH)' union all select 'features', count(*) from features.complaint_features where batch_month = '$(MONTH)'"

bad-batch:  ## feed a broken copy of 2026-05; validate must fail
	uv run python -m complaint_routing.fixtures
	$(AIRFLOW) dags trigger complaints_etl --conf '{"month": "2026-05", "raw_path": "data/fixtures/bad_batch_2026-05.parquet"}'

## --- feature store (Feast, separate env in feature_repo/) ---
feast-apply:
	cd feature_repo && uv run feast apply

feast-demo:  ## point-in-time retrieval between two feature rows returns the earlier one
	cd feature_repo && uv run python point_in_time_demo.py

feast-materialize:  ## latest row per company into the online store (read by serving in Lab 4)
	cd feature_repo && uv run feast materialize 2022-01-01T00:00:00 2026-08-01T00:00:00

## --- training, registry, reproduction (MLflow) ---
population-mix:
	uv run python -m complaint_routing.population

train:  ## baselines + candidate, logged with data version and commit; registers the candidate as champion
	uv run python -m complaint_routing.train

reproduce:  ## rebuild the champion from its two tags and compare the metric
	uv run python -m complaint_routing.reproduce --model complaint-router --alias champion

.PHONY: setup lint test up down data-pull data-push data-rebuild etl-backfill etl-month etl-wait counts bad-batch feast-apply feast-demo feast-materialize population-mix train reproduce
