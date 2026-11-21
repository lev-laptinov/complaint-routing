-include .env
export
# progress lines of long LLM runs appear as they happen
export PYTHONUNBUFFERED = 1

MONTH ?= 2026-05
# The LLM server reads its model name from config/llm.yaml, so serving and evaluation never drift.
LLM_MODEL := $(shell sed -n 's/^  model: *//p' config/llm.yaml | head -1)
export LLM_MODEL
VLLM_VERSION = v0.30.0
VLLM_SRC ?= $(HOME)/src/vllm
PARAM ?= retrieval.chunk_size
VALUES ?= 60,120,240
AIRFLOW = docker compose exec -T airflow airflow
PSQL = docker compose exec -T postgres psql -U $(POSTGRES_USER)
# Lab 4: the prediction log and labels live in the warehouse database (schema serving)
WAREHOUSE_DSN ?= postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:5433/warehouse
export WAREHOUSE_DSN
PROFILE ?= complaint-routing
K = kubectl --context $(PROFILE) -n complaint-routing
ROWS ?= 300
SEED ?= 2026
VERSION ?=

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

## --- Lab 3: LLM baseline and retrieval ---
llm-image:  ## vLLM CPU image from source (published ARM64 images exit 132 on Apple Silicon); stop airflow+mlflow first: 4 parallel jobs ran the 8 GB Docker VM out of memory
	[ -d $(VLLM_SRC) ] || git clone --depth 1 --branch $(VLLM_VERSION) https://github.com/vllm-project/vllm.git $(VLLM_SRC)
	cd $(VLLM_SRC) && docker build -f docker/Dockerfile.cpu --platform=linux/arm64 --build-arg max_jobs=2 \
		--tag vllm-cpu-arm64:0.30.0 --target vllm-openai .

llm-up:  ## OpenAI-compatible server on localhost:8000 with the model named in config/llm.yaml
	docker compose --profile llm up -d vllm
	@until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done; curl -s localhost:8000/v1/models; echo

llm-down:
	docker compose --profile llm stop vllm

corpus:  ## re-fetch the Ask CFPB snapshot (the committed file is the corpus version; this changes it)
	uv run python -m complaint_routing.llm.corpus

rag-index:  ## corpus -> chunks -> embeddings -> pgvector, for the settings in config/llm.yaml
	uv run python -m complaint_routing.llm.index

rag-ask:  ## make rag-ask Q="Can a debt collector call me at work?"
	uv run python -m complaint_routing.llm.rag "$(Q)"

llm-baseline:  ## zero-shot queue prediction, logged next to the Lab 2 runs with the same metric
	uv run python -m complaint_routing.llm.classify

rag-eval:  ## retrieval + answer metrics on eval/rag_questions.jsonl, one MLflow run
	uv run python -m complaint_routing.llm.evaluate

rag-sweep:  ## one parameter, several values: make rag-sweep PARAM=retrieval.chunk_size VALUES=60,120,240
	uv run python -m complaint_routing.llm.evaluate --sweep $(PARAM)=$(VALUES)

## --- Lab 4: serving on Minikube, monitoring and drift ---
k8s-up:  ## ONE COMMAND: storage, Minikube, images (tag = commit), every workload, smoke request
	scripts/k8s-up.sh

k8s-forward:  ## 127.0.0.1 access: router 8081, assistant 8082, Grafana 3000, Prometheus 9090, Alertmanager 9093, MLflow (cluster) 5051
	scripts/k8s-forward.sh

k8s-forward-stop:
	scripts/k8s-forward.sh stop

k8s-status:
	$(K) get pods,svc,cronjob,jobs -o wide

k8s-down:  ## delete the cluster (compose storage and the registry stay)
	minikube delete -p $(PROFILE)

smoke:  ## the smoke request from the host (needs make k8s-forward)
	uv run python -m complaint_routing.serving.smoke http://localhost:8081

replay:  ## ROWS held-out complaints, unchanged, through POST /predict (needs make k8s-forward)
	uv run python -m complaint_routing.monitoring.replay --rows $(ROWS) --seed $(SEED) --labels

replay-shift:  ## the documented shift: 30 % of narratives cut to their first 25 words
	uv run python -m complaint_routing.monitoring.replay --rows $(ROWS) --seed $(SEED) --shift truncate --fraction 0.3 --words 25

drift-now:  ## run the drift CronJob once, now, and print its log
	$(K) delete job drift-manual --ignore-not-found
	$(K) create job drift-manual --from=cronjob/drift-monitor
	$(K) wait --for=condition=complete job/drift-manual --timeout=300s
	$(K) logs job/drift-manual | grep -v -i warn

psql-window:  ## the newest rows of the prediction log
	@$(PSQL) -d warehouse -c "select ts, model_version as v, predicted_queue, round(max_proba::numeric, 2) as p, narrative_len, left(narrative, 70) as narrative from serving.predictions order by ts desc limit 20"

registry:  ## versions and aliases of complaint-router
	uv run python -m complaint_routing.registry show

promote:  ## make promote VERSION=2: move @champion (the old one becomes @previous), then restart the router
	uv run python -m complaint_routing.registry promote --version $(VERSION)
	$(K) rollout restart deployment/router
	$(K) rollout status deployment/router --timeout=300s

reference:  ## backfill the drift reference of the current champion (train.py logs it for new versions)
	uv run python -m complaint_routing.monitoring.reference --alias champion

dashboard:  ## regenerate k8s/monitoring/dashboard.json
	uv run python scripts/make_dashboard.py

.PHONY: k8s-up k8s-forward k8s-forward-stop k8s-status k8s-down smoke replay replay-shift drift-now psql-window registry promote reference dashboard
.PHONY: llm-image llm-up llm-down corpus rag-index rag-ask llm-baseline rag-eval rag-sweep
.PHONY: setup lint test up down data-pull data-push data-rebuild etl-backfill etl-month etl-wait counts bad-batch feast-apply feast-demo feast-materialize population-mix train reproduce
