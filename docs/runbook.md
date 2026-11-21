# Runbook: complaint routing alerts

Who receives: the ML engineer on call (the project owner), by e-mail from Alertmanager
(subject `[complaint-routing] FIRING: <alert> (<severity>)`). `page` alerts need a look within the
hour during working hours; `ticket` alerts within a working day. Where everything is
(after `make k8s-forward`): Grafana <http://localhost:3000> (dashboard "Complaint routing"),
Prometheus <http://localhost:9090/alerts>, Alertmanager <http://localhost:9093>, MLflow
<http://localhost:5050> (experiment `complaint-routing-monitoring`: one run per drift window).

## InputDriftNarrativeLength

The narrative-length distribution of the last window differs from the reference window
(validation months 2026-03..04) with KS p < 0.001 on up to 300 rows. On unshifted months this
happened in 0 of 60 windows (design 6.6), so treat it as real until shown otherwise.

First ten minutes:
1. Grafana, row "Drift": is it one window or several in a row? Do `n_words`, `oov_share`,
   `mask_share` move with it? Is `max_proba` (confidence) dropping?
2. Open the MLflow run of the window (`complaint-routing-monitoring`, newest run, artifact
   `evidently_drift_report.html`): which way did the length move (shorter or longer)?
3. Look at a few rows of the window:
   `make psql-window` (or `select narrative_len, left(narrative, 120) from serving.predictions
   order by ts desc limit 20`).
4. Decide, and write the decision in the alert's e-mail thread:
   - **Upstream change** (intake form cut the text box, a new channel, a broken scrubber that
     masks too much): tell the intake team; the model is not the problem; retraining will not help.
   - **Real change in complaints** (new product, new wording): the retraining run the alert
     already started (stretch goal, Job `retrain-<timestamp>`) registers a `candidate`; compare
     it with the champion (`make registry`, its `promotion_gate` tag) before `make promote`.
   - **Replay or test traffic**: silence the alert in Alertmanager with a reason.

## DriftMonitorStale

No drift window for 15 minutes: the CronJob is not running or fails before pushing.
`make k8s-status` (last Jobs), `kubectl -n complaint-routing logs job/<last drift job>`.
Typical causes: the registry (MLflow) is down, the reference artifact is missing
(`make reference`), the database is unreachable.

## PredictionsNotLogged

The router served more than 10 predictions in 10 minutes but the drift job found none in the
prediction log: monitoring is blind. Check `router_prediction_log_errors_total` (Grafana) and the
router's log (`kubectl -n complaint-routing logs deploy/router | grep "prediction log"`). Usual
cause: PostgreSQL unreachable or the `serving` schema dropped. The router keeps routing; fix the
log first, because no drift alert can fire while it is empty.

## RoutingQualityDrop

Macro-F1 on predictions whose labels arrived (last 24 h, >= 1,000 labels) is below 0.70
(test 0.776). Labels arrive 15-60 days late, so this is the slow signal: compare with the drift
panels of the weeks the complaints were received; open the confusion matrix by queue before
deciding on retraining.

## RouterNotReady

No ready router pod for 5 minutes; the intake falls back to the general triage queue.
`curl localhost:8081/health/ready` (or `kubectl -n complaint-routing describe pod -l app=router`)
says what it waits for: `model` (registry down, alias missing or pointing at a version whose
artifacts are gone; roll back with `make promote VERSION=<previous>`), `features` or
`prediction_log` (PostgreSQL).

## RouterLatencyHigh

p95 above 300 ms for 10 minutes. Compare request rate (more traffic?), the pod's CPU throttling
(`kubectl top pod`), and the prediction-log insert time (PostgreSQL slow?).

## RouterErrors

More than 1 % of requests fail for 10 minutes. The router's log has the stack trace; a new model
version that fails on some inputs is rolled back with `make promote VERSION=<previous>`.

## AssistantNotReady

The analyst assistant is not ready for 10 minutes: usually the LLM server outside the cluster
(vLLM, `make llm-up`) is down or serves another model than `config/llm.yaml` names, or the
retrieval index for the configured settings is missing (`make rag-index`).
