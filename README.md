# Complaint routing — keeping a text classifier up to date

A consumer-finance complaint intake service has to send every free-text complaint to the right product review queue, and misrouted complaints wait in the wrong queue while the 15-day response clock runs.
The data behind it moves: the CFPB product taxonomy changed in 2017 and 2023–2024, credit reporting grew from about 20 % to about 90 % of complaints, and on 2026-08-14 the CFPB stopped publishing complaint narratives.
This project builds a system that classifies each complaint narrative into a routing queue at intake, trained on monthly batches of CFPB complaints replayed in time order.
A new model reaches production only after it beats the current one on the most recent held-out months, and drift in language, queue mix and upstream schema is monitored before labels arrive 15–60 days later.
The design is in [docs/design.md](docs/design.md); the course builds it over five labs (pipeline, LLM assistant, serving and monitoring, agent).

## Quick start

```bash
make setup   # pinned environment (uv), pre-commit hook, local .env from .env.example
make lint    # ruff, the same check CI runs on every push
make test    # unit tests of the cleaning rules, the quality gate and the point-in-time feature
make up      # PostgreSQL 16, MinIO, MLflow 3.16.1, Airflow 3.3.1 (docker compose)
```

| Service | URL (localhost only) | What it holds |
|---|---|---|
| Airflow 3.3.1, LocalExecutor | <http://localhost:8080> | DAG `complaints_etl` (no login: local simple auth manager) |
| MLflow 3.16.1 | <http://localhost:5050> | runs (PostgreSQL db `mlflow`), artifacts (MinIO bucket `mlflow`), registry |
| MinIO (`pgsty/silo:RELEASE.2026-09-16T00-00-00Z`) | API <http://localhost:9310>, console <http://localhost:9311> | bucket `dvc` (DVC remote), bucket `mlflow` |
| PostgreSQL 16 | `localhost:5433` | databases `airflow`, `mlflow`, `warehouse` (schemas `raw`, `dwh`, `features`, `feast_online`) |

Credentials live in `.env` (copied from `.env.example`, never committed). The Makefile exports them, so every command below works as written.

## Lab 2

### 1. Raw data, versioned with DVC

- **Source:** the CFPB Consumer Complaint Database narratives archive (FOIA reading room), 18 ZIP exports covering 2022-01 → 2026-07, 1.1 GB zipped, downloaded on 2026-10-24.
- **Sample (the archive is far above 200 MB):** for each month, take the complaints that have a published narrative, drop duplicate `Complaint ID`s, sort by `Complaint ID`, and draw 1,500 rows with `pandas.DataFrame.sample(random_state=YYYYMM)`. Columns are kept exactly as delivered. 55 files `data/raw/complaints_YYYY-MM.parquet`, 82,500 rows. Code: [`ingest.py`](src/complaint_routing/ingest.py).
- **Remote:** S3-compatible MinIO, `s3://dvc/complaint-routing` at `http://localhost:9310`. The remote config is committed ([`.dvc/config`](.dvc/config)); the keys come from `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `.env`.

Retrieve the data on a clean clone:

```bash
make setup && make up && make data-pull
uv run dvc status data/raw.dvc      # -> "Data and pipelines are up to date."
```

What arrives must match [`data/raw.dvc`](data/raw.dvc): **md5 `835e7d9b012a06bf4e2bddb624fdaf4f.dir`, 55 files, 26,111,493 bytes**. This md5 is the data version every pipeline row and every MLflow run carries.

The remote is a container, so on a different machine its bucket starts empty. Then rebuild the files from the public archive and seed your remote; the build is deterministic (two builds gave identical md5 for all 55 files), and `--verify` runs `dvc status` against the committed hash:

```bash
make data-rebuild && make data-push
```

### 2. ETL pipeline: `complaints_etl`

[`dags/complaints_etl.py`](dags/complaints_etl.py): `extract → validate → transform → load → build_features → quality_check`.

![successful run](docs/screenshots/airflow_success_graph.png)

- **The batch is the data interval, never the clock.** Airflow 3 no longer creates intervals for `@monthly` (start == end), so the DAG declares `schedule=CronDataIntervalTimetable("@monthly", timezone="UTC")`: a run for logical date 2026-05-01 has the interval [2026-05-01, 2026-06-01) and processes month `2026-05`. A manual run has no interval in Airflow 3, so it must pass `{"month": "YYYY-MM"}` in the conf, or it fails with that message.
- **Run a specific interval:** `make etl-month MONTH=2026-05` (an Airflow backfill of exactly that interval; reruns it even if it succeeded). All months in order: `make etl-backfill` (55 runs, `max_active_runs=1`, because the features of month M read months M-5..M-1).
- **Layers in PostgreSQL `warehouse`:** `raw.complaints` (as delivered + `data_version`), `dwh.complaints` (typed, cleaned, mapped to a queue), `features.complaint_features`.

**Idempotency.** `load` and `build_features` delete the batch month and insert it again in one transaction ([`warehouse.replace_month`](src/complaint_routing/warehouse.py)). The same interval, run twice:

```text
### run 1: make etl-month MONTH=2026-05
backfill__2026-06-01T00:00:00+00:00|success|2026-05-01 00:00:00+00|2026-06-01 00:00:00+00
  layer   | count
----------+-------
 raw      |  1500
 dwh      |  1411
 features |  1411

### run 2: make etl-month MONTH=2026-05
backfill__2026-06-01T00:00:00+00:00|success|2026-05-01 00:00:00+00|2026-06-01 00:00:00+00
  layer   | count
----------+-------
 raw      |  1500
 dwh      |  1411
 features |  1411
```

The whole warehouse stays at 82,500 raw / 71,419 dwh / 71,419 feature rows after both runs. 1500 → 1411: `transform` removed 89 duplicate narratives (the same text filed more than once) in that month.

**Quality gate.** `validate` checks the raw batch against its contract before anything is written: schema (16 columns), ≥ 500 rows, ≤ 5 % null narratives, no product missing from [`config/queue_mapping.csv`](config/queue_mapping.csv), every `Date received` inside the batch month, no duplicate ids, and not the same file as the previous month (stale partition). `quality_check` checks the warehouse after the load: rows equal to what `transform` produced, features for every row, no row outside the month, one `feature_version`.

How it was shown to fail: `make bad-batch` writes [a broken copy of 2026-05](src/complaint_routing/fixtures.py) (300 narratives set to null, 50 rows with the product `Crypto lending`, 100 rows dated 2026-04-30) and triggers the DAG on it. `validate` fails on all three, the downstream tasks never run, and the 2026-05 counts above stay 1500 / 1411 / 1411.

![failed validate](docs/screenshots/airflow_failed_validate.png)

### 3. Feature table and where each feature comes from at serving time

`build_features` fills `features.complaint_features` (one row per complaint, `event_timestamp` = date received, column `feature_version = 'v1'`) with **`company_queue_mix`**: the share of each of the 10 queues among the company's complaints received in [d − 150 days, d − 60 days), plus their count `mix_n`. The 60-day lag is the label delay: a complaint's final queue is known only when it closes. The serving-time source of every feature is the table in [design 5.3](docs/design.md#53-techniques). Training builds exactly the same table from the DVC files; checked: all 71,419 rows match the warehouse, max absolute difference of the mix features 0.0.

**Feast (optional point).** [`feature_repo/`](feature_repo) has its own environment (Feast 0.66 needs pandas < 3). Entity `company`, feature view `company_queue_mix`, offline store = the PostgreSQL table above, online store = schema `feast_online`.

```bash
make feast-apply && make feast-demo
```

```text
two feature rows of CAPITAL ONE FINANCIAL CORPORATION:
event_timestamp  mix_card  mix_checking_savings  mix_n
     2024-03-02  0.453333              0.146667     75
     2024-03-05  0.460526              0.144737     76

get_historical_features at 2024-03-03 12:00:00 (between the two rows):
                          company     event_timestamp  mix_card  mix_checking_savings  mix_n
CAPITAL ONE FINANCIAL CORPORATION 2024-03-03 12:00:00  0.453333              0.146667     75

OK: Feast returned the earlier row; the later one is after the entity timestamp.
```

`make feast-materialize` loads the latest row per company into the online store (19,052 rows) for the Lab 4 service.

### 4. Baselines and model on a time split

Train 2022-01 → 2026-02 (64,269 rows), validation 2026-03 → 04 (2,856), test 2026-05 → 07 (4,294); why not a random split: [design 5.4](docs/design.md#54-experimentation-and-validation). `make train`.

| Test 2026-05 → 07 | macro-F1 | accuracy | min recall over queues | accuracy re-weighted to the full intake mix |
|---|---|---|---|---|
| Majority queue (always `credit_reporting`) | 0.008 | 0.042 | 0.000 | **0.914** |
| Keyword rules, fallback to majority | 0.524 | 0.523 | 0.186 | 0.764 |
| **TF-IDF + one-hot + company mix → logistic regression** | **0.776** | **0.830** | 0.270 (`debt_management`) | 0.869 |

The trained model beats both rules on the metric the design optimizes (macro-F1). On the last column the majority rule wins: 91.4 % of all complaints in the test months are credit reporting, so "always credit reporting" is right 91 % of the time and routes nothing else. That is why accuracy is not a target ([design 3](docs/design.md#3-success-metrics)). Every queue except `debt_management` (37 test rows) has recall ≥ 0.71; that one queue misses the 0.60 recall target.

### 5. Experiment tracking, registry and reproduction

MLflow runs as a server with a PostgreSQL backend store and MinIO artifacts (`docker compose`). Every run logs its parameters, metrics and tags `data_version` (DVC md5 of `data/raw`), `git_commit`, `git_dirty`, `feature_version`, `mapping_version`. The candidate is registered as **`complaint-router` version 1**, alias **`champion`** (Lab 4 loads `models:/complaint-router@champion`), tagged with both lineage identifiers:

![registered version](docs/screenshots/mlflow_champion_v1.png)

One command reproduces the champion:

```bash
make reproduce
```

It reads `git_commit` and `data_version` from the registered version, checks the commit out into a temporary git worktree, `dvc pull`s that commit's data from the remote, trains there with the logged parameters in an environment built from that commit's `uv.lock`, and compares test macro-F1. **Tolerance: ±0.005 macro-F1**; it covers BLAS/thread scheduling differences between machines, while the data, code, environment and seed are fixed. Output:

```text
$ git worktree add --detach /var/folders/.../repro-2ys05gmt 4a2c1f3fb63aa9fdea78ecf2ca5fa49b0041c05c
$ uv run --frozen dvc remote modify --local minio endpointurl http://localhost:9310
$ uv run --frozen dvc pull data/raw.dvc
$ uv run --frozen python -m complaint_routing.train --no-register --only-candidate ... --c 4.0 --max-features 100000 --min-df 3 --seed 42
$ git worktree remove --force /var/folders/.../repro-2ys05gmt

test_macro_f1: registered 0.7762, reproduced 0.7762 (run bb8664305a5c42f7a10608672e2ed3e7)
|diff| 0.0000 <= tolerance 0.005 -> REPRODUCED
```

The remote address is taken from today's `.dvc/config`, the data hash from the commit: the remote moved from port 9010 to 9310 after version 1 was trained, and the data it pulls is still identified by the same md5.

### Hardware and timing

MacBook Pro M1 Pro, 16 GB, Docker Desktop 28.3.2 (8 GB for containers). Raw build 1.5 min; one DAG run ≈ 6 s, the 55-month backfill ≈ 7 min; training ≈ 8 min on CPU; `make reproduce` ≈ 9 min (it also builds a fresh environment).

## Lab 3

The LLM component has two jobs ([design 5.1](docs/design.md#51-problem-statement)): an **LLM baseline for the routing task itself** (zero-shot queue prediction, scored with the Lab 2 metric next to the Lab 2 runs) and an **analyst assistant** that answers questions about the rules behind a complaint from the CFPB's own consumer guidance, with citations. Every setting is in [`config/llm.yaml`](config/llm.yaml); every prompt is a file in [`prompts/`](prompts) whose sha256 is logged on every run.

```bash
make llm-image      # once: build the vLLM 0.30.0 CPU image from source (1.5–2 h on an M1 Pro)
docker compose stop airflow   # the 8 GB Docker VM cannot hold Airflow and vLLM at once
make llm-up         # vLLM OpenAI-compatible server on localhost:8000, model from config/llm.yaml
make rag-index      # corpus -> chunks -> embeddings -> pgvector; prints the document count
make rag-ask Q="Can a debt collector tell my employer about my debt?"
make llm-baseline   # zero-shot routing on 300 test complaints, logged in experiment complaint-routing
make rag-eval       # retrieval + answer metrics on 57 questions, experiment complaint-assistant-rag
make rag-sweep PARAM=retrieval.chunk_size VALUES=60,120,240
```

| Service | URL (localhost only) | What it holds |
|---|---|---|
| vLLM 0.30.0 (CPU, image `vllm-cpu-arm64:0.30.0`) | <http://localhost:8000/v1> | `Qwen/Qwen2.5-0.5B-Instruct`, fp16, OpenAI-compatible API |
| PostgreSQL 16 + pgvector 0.8.7 (`pgvector/pgvector:0.8.7-pg16-trixie`) | `localhost:5433` | schema `rag` in database `warehouse`: `rag.indexes`, `rag.chunks` |

The PostgreSQL image changed from `postgres:16` to the pgvector build of the same major version and the same Debian release, so the Lab 2 volume is reused as is (82,500 raw rows still there after the switch).

### 1. Corpus

- **Ask CFPB**, the CFPB's consumer Q&A (consumerfinance.gov/ask-cfpb): every English answer page listed in the site's `sitemap.xml`, fetched on 2026-11-07. One page = one document, id `ask-<n>` from the URL. Categories mirror the routing queues (mortgages 191 pages, bank accounts 84, credit reports 55, auto loans 54, student loans 53, credit cards 51, debt collection 47, prepaid cards 33, fraud and scams 28, payday loans 19, reverse mortgages 18, money transfers 10, general 13).
- **656 documents, 211,230 words** (median 261 words per page); 17 pages whose answer is shorter than 200 characters are dropped. Public: a US federal government work.
- The snapshot is committed, [`data/corpus/ask_cfpb.jsonl`](data/corpus/ask_cfpb.jsonl) (1.5 MB), **md5 `f9e1907f4bfe9d6b98e28f0678efe7ad`**; this md5 is the corpus version on every run. `make corpus` re-fetches the live site, which produces a new version ([`corpus.py`](src/complaint_routing/llm/corpus.py)).

### 2. Retrieval pipeline

Diagram: [design 6.1, Retrieval pipeline](docs/design.md#retrieval-pipeline-lab-3). Chunk size, overlap, embedding model and top-k live only in `config/llm.yaml`. The index rows are keyed by an **index version** = hash(corpus md5, chunk size, overlap, embedding model), so the query path refuses to run against an index that does not match the config and the corpus, and sweeps build their own index versions next to the served one.

```text
$ make rag-index
corpus data/corpus/ask_cfpb.jsonl (md5 f9e1907f4bfe9d6b98e28f0678efe7ad), chunk_size 120, overlap 20, embedding sentence-transformers/all-MiniLM-L6-v2
index 6c1f3b1fc70f: 656 documents, 2300 chunks -> rag.chunks
```

**The index has 656 documents and 2,300 chunks** (57 s on the laptop CPU, embeddings included).

The query path end to end (`make rag-ask`): distance, chunk id and title of the top 5, then the answer, tokens, latency and the answer prompt's hash.

```text
$ make rag-ask Q="Can a debt collector tell my employer about my debt?"
  0.178  ask-332#3      Can debt collectors tell other people, like family, friends, or my employer, about my debt?
  0.252  ask-1411#6     How do I get a debt collector to stop calling or contacting me?
  0.267  ask-1699#3     How do I tell if a debt collector is legitimate or a scam?
  0.267  ask-1699#0     How do I tell if a debt collector is legitimate or a scam?
  0.271  ask-332#0      Can debt collectors tell other people, like family, friends, or my employer, about my debt?

Yes, a debt collector can tell your employer about your debt. If a debt collector calls your employer, you might want to talk to the person who took the call to find out what the debt collector said. A debt collector may not tell your employer that you owe a debt. If the debt collector has told your employer that you owe a debt, you may want to speak to an attorney about your rights.

[851 tokens in, 83 out, 7.2 s, prompt e749481e4147]
```

Retrieval is right (the relevant page is first), the answer opens with the opposite of what the page says and then corrects itself, and it cites nothing. This is the failure pattern the evaluation below measures.

### 3. LLM baseline for routing (same task, same metric)

`make llm-baseline` sends 300 test complaints (seeded random sample of 2026-05 → 07, natural queue mix) to the LLM with the prompt in [`prompts/classify_v2.txt`](prompts/classify_v2.txt), and scores the champion on exactly the same rows. Both runs are in the Lab 2 experiment `complaint-routing`, with the same metric names as the Lab 2 runs, so they sort side by side in the MLflow UI. All runs below: commit `369aeba`, `git_dirty=false`.

| Run (experiment `complaint-routing`) | prompt hash | macro-F1 | accuracy | min recall | unparsed | p50 / p95 latency |
|---|---|---|---|---|---|---|
| `llm-zero-shot-Qwen2.5-0.5B-Instruct`, prompt v1 (`1974752…`) | `4876df1eeccd` | 0.040 | 0.083 | 0.000 | 0.7 % | 0.93 / 1.43 s |
| `llm-zero-shot-Qwen2.5-0.5B-Instruct`, **prompt v2** (`eea0fc8…`) | `860affc491b5` | **0.339** | 0.330 | 0.000 | 5.7 % | 1.02 / 1.52 s |
| `champion-on-llm-sample` (`complaint-router@champion`, same 300 rows) | — | **0.829** | 0.850 | 0.588 | — | (ms, in process) |

- **Prompt v1 collapsed:** the 0.5B model answered `credit_reporting` for 273 of 300 complaints. v2 puts the complaint first and the options after it, with the examples of each product next to its name; macro-F1 went from 0.04 to 0.34. v1 is a file in the repository with its own hash: its run stays reproducible (every rerun of v1 gave 0.0405 again, at temperature 0), and `config/llm.yaml` now points to v2.
- **The decision this comparison informs:** a zero-shot 0.5B LLM is not a replacement for the trained classifier (0.34 vs 0.83 macro-F1 on the same rows) and not even a fallback for a new queue. The routing service stays on the champion; the LLM's job in this system is the assistant below.
- v1's "accuracy re-weighted to the intake mix" is 0.919, only 1 pp below the champion's 0.928: a model that almost always says credit reporting looks good on that number, which is why it is not a target (design 3).

### 4. Evaluation set

[`eval/rag_questions.jsonl`](eval/rag_questions.jsonl): **57 questions** over all 10 queues, each with 1–2 relevant document ids and a reference answer. They were **generated with an LLM (Claude) from the corpus, then read and corrected against the documents**. No question quotes its document: the questions describe the complaint situation ("the agency told her boss…") instead of using the page title or product names; a check for 3-word phrases shared with the document title found only generic ones ("is there a"). Relevance is per document, so the chunk-size sweep is scored against the same labels. Details: [design 5.4](docs/design.md#54-experimentation-and-validation).

### 5. Retrieval and answer metrics, one parameter at a time

`make rag-sweep PARAM=retrieval.chunk_size VALUES=120,60,240`; overlap (20 words), embedding model (`all-MiniLM-L6-v2`), top-k (5), LLM and prompts unchanged. Experiment `complaint-assistant-rag`, commit `369aeba`.

| chunk size (words) | chunks | hit rate@1 | hit rate@5 | hit rate@10 | MRR@10 | **answer correct (hand rubric, 19 q)** | answer correct (LLM judge v1, 57 q) | judge agrees with hand | tokens in / out per answer | p50 / p95 latency | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 5,275 | 0.351 | 0.649 | 0.807 | 0.507 | **0.421** (8/19) | 0.807 | 0.53 | 597 / 90 | 4.4 / 11.2 s | `90ba3a1f` |
| **120 (served)** | 2,300 | **0.491** | **0.684** | 0.807 | 0.571 | 0.263 (5/19) | 0.842 | 0.37 | 854 / 85 | 5.1 / 13.5 s | `e453e367` |
| 240 | 1,238 | 0.491 | 0.667 | **0.842** | **0.572** | 0.211 (4/19) | 0.842 | 0.26 | 1,137 / 75 | 4.9 / 11.4 s | `24ac1bd2` |

A second parameter, the embedding model (retrieval only, chunk size 120; `make rag-sweep PARAM=retrieval.embedding_model VALUES=…`):

| embedding model | hit rate@1 | hit rate@5 | hit rate@10 | MRR@10 | run |
|---|---|---|---|---|---|
| **`sentence-transformers/all-MiniLM-L6-v2` (served)** | **0.491** | 0.684 | 0.807 | **0.571** | `569f6e5e` |
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | 0.404 | 0.649 | 0.772 | 0.514 | `933d9492` |
| `BAAI/bge-small-en-v1.5` | 0.421 | **0.772** | **0.860** | 0.559 | `7bf09278` |

What the tables say:
- **Chunk size.** 60-word chunks split an answer into pieces that each match the question less well: hit rate@1 drops from 0.49 to 0.35. Going from 120 to 240 words changes little for retrieval (hit@10 +0.035, the same MRR) but makes every prompt 33 % longer. 120 stays the served value.
- **The answers move the other way.** The small model answers better from shorter context (8/19 correct at 60 words vs 5/19 at 120 and 4/19 at 240): with 1,100 tokens of excerpts it more often copies an unrelated passage. On 19 graded questions the difference is within noise (±0.11 standard error), so the next step is to hand-grade all 57 before changing the served value.
- **Embedding model.** `bge-small-en-v1.5` finds the right document somewhere in the top 5 more often (0.77 vs 0.68) but ranks it first less often; with top-k = 5 it would give the LLM the right page for 5 more of the 57 questions. It is the first candidate for the next sweep (with its query prefix).
- **Hit rate@5 of 0.68 with MRR 0.57**: when the retriever finds the page it is usually first or second. Of the 18 misses at top 5, some are unlabelled pages that answer the question too: q46 (servicer ignored an error notice) retrieves "How do I dispute an error or request information about my mortgage?" first, q35 (is a prepaid balance insured) retrieves the page on registering a prepaid card, which is the condition for FDIC insurance. So the labels undercount hit rate somewhat; adding second relevant documents is a change to the evaluation set, i.e. a new `eval_set_md5`. Others are real misses: q40 (when private mortgage insurance ends) retrieves a reverse-mortgage page.

**The answer metric is a hand rubric, because the LLM judge does not work at 0.5B.**
- The rubric: an answer is correct if a reader would learn the key fact of the reference answer, without a contradiction. Applied to every third question (19 per configuration), stored in [`eval/hand_grades.csv`](eval/hand_grades.csv) with a one-line reason for each wrong answer. Each grade is keyed by the sha256 of the answer text, so `evaluate` counts it only for that exact answer; the final runs regenerated identical answers (temperature 0) and matched all 19 grades.
- **Judge v1** ([`prompts/judge_v1.txt`](prompts/judge_v1.txt), hash `7b1a3fbaa9d8`) says YES to almost everything: it agrees with the hand grades on 22 of 57 graded answers (39 %), with 33 false YES and 2 false NO. A typical false YES: "Yes, a debt collector can threaten to arrest someone" graded YES against the reference "a collector cannot threaten arrest".
- **Judge v2** ([`prompts/judge_v2.txt`](prompts/judge_v2.txt), hash `025186aeb8dc`, stricter, two worked examples) says NO to all 57 answers (run `8a29e7fa`). With identical answers, `answer_correct` went from 0.84 to 0.00 by editing the judge prompt alone — the reason the judge prompt is versioned and hashed like every other prompt.
- Also measured on every run: citation rate 0–3.5 % (the 0.5B model ignores the "cite [ask-id]" instruction), so `cited_outside_retrieved_rate` (0 %) is not yet informative; refusal rate 0 % except one refusal at chunk size 60 when the right page was at rank 1.

### 6. Cost and latency; local inference vs an API

Measured on the runs above (MacBook Pro M1 Pro, 8 cores, 16 GB; Docker VM 8 CPUs / 8 GB; vLLM container 4.9 GB resident, `Qwen2.5-0.5B-Instruct` in fp16, no quantization, no GPU):

| Call | calls per run | tokens in / out per call | p50 | p95 | cold first call | compute per call |
|---|---|---|---|---|---|---|
| Assistant answer (chunk 120, top-5) | 57 | 854 / 85 | 5.1 s | 13.5 s | 2.8 s | 5.9 s |
| Judge | 57 | 240 / 2 | 0.61 s | 0.98 s | 0.62 s | 0.65 s |
| Routing (prompt v2) | 300 | 418 / 3.4 | 1.02 s | 1.52 s | 0.98 s | 1.05 s |

- **Money:** none beyond the laptop; the cost is the local resource: about 6 s of the 8-core Docker VM per answered question and 4.9 GB of its memory while the server runs. A whole 57-question evaluation takes 6 min of LLM compute plus 37 s of judging.
- **Cold prompt:** each run starts right after `docker compose restart vllm`, so its first call has an empty prefix cache (`llm_latency_first_s`). It is not slower than the median here, because every answer prompt carries different excerpts: only the ~100-token instruction prefix can be cached.
- **The tail** (p95 13.5 s vs p50 5.1 s) comes from output length: the run produced 4,837 output tokens in 338 s including prefill (about 14 tokens/s), and a few answers run to the 256-token limit (numbered step lists). Prompt length matters less: 240-word chunks add 280 input tokens per answer and do not move p50.
- **Why local and not an API** ([design 6.3, 6.7, 7.1](docs/design.md#67-cost)): the analyst questions and, in the routing baseline, complaint narratives never leave the machine; the model behind the name cannot change under us, so runs stay reproducible; no key to protect. What it costs us: answer quality (5 of 19 correct, a judge that cannot judge), 5–13 s per answer, and 4.9 GB of an 8 GB Docker VM — the first vLLM image build with 4 parallel jobs ran the VM out of memory and killed Airflow and MLflow (exit 137), so `make llm-image` now uses 2 jobs. Moving to an API is three lines of `config/llm.yaml` and a key in `.env`.

### Hardware and timing (Lab 3)

vLLM image build from source: about 1.5–2 h in total on the M1 Pro (ARM Compute Library, oneDNN and vLLM's CPU kernels), 5.2 GB image. Index build: 57 s. A 57-question evaluation with judge: about 7 min; the 300-complaint routing baseline: about 6 min. Airflow is stopped while vLLM runs (`docker compose stop airflow`): the Docker VM does not hold Airflow, MLflow, PostgreSQL and vLLM at once.

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
|---|---|---|
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | `docs/design.md` follows the template, every section filled or marked as an assumption | Lab 1 |
| Data versioned | `make data-pull` retrieves `data/raw` and `dvc status` matches md5 `835e7d9b…dir` ([1](#1-raw-data-versioned-with-dvc)) | Lab 2 |
| Pipeline idempotent | the same interval run twice gives 1500 / 1411 / 1411 rows both times, pasted in [2](#2-etl-pipeline-complaints_etl) | Lab 2 |
| Quality gate present | `make bad-batch` fails `validate` on three named rules; [screenshot](docs/screenshots/airflow_failed_validate.png) | Lab 2 |
| Experiments tracked | every run in experiment `complaint-routing` has params, metrics, `data_version` and `git_commit` ([5](#5-experiment-tracking-registry-and-reproduction)) | Lab 2 |
| Model registered with lineage | `complaint-router` v1 `@champion` tagged with data version and commit; `make reproduce` gives the same macro-F1 | Lab 2 |
| LLM component evaluated | `eval/rag_questions.jsonl` (57 questions); `make rag-eval` / `make rag-sweep` log hit rate@k, MRR@10, the hand-rubric and judge answer metrics as runs in `complaint-assistant-rag`; `make llm-baseline` logs macro-F1 next to the Lab 2 runs ([3](#3-llm-baseline-for-routing-same-task-same-metric), [5](#5-retrieval-and-answer-metrics-one-parameter-at-a-time)) | Lab 3 |
| Prompts versioned | every prompt is a file in `prompts/`; its sha256 (`prompt_hash*` tags) and the config hash are on every LLM run; changing `judge_v1` → `judge_v2` alone moved `answer_correct` 0.84 → 0.00 on identical answers | Lab 3 |
| Cost and latency recorded | every LLM run logs tokens in/out, compute seconds, cold first-call latency, p50 and p95 (answers: 854 / 85 tokens, p50 5.1 s, p95 13.5 s, 5.9 s of local compute) ([6](#6-cost-and-latency-local-inference-vs-an-api)) | Lab 3 |
