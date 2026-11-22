"""Writes k8s/monitoring/dashboard.json, the Grafana dashboard "Complaint routing" (provisioned).

    uv run python scripts/make_dashboard.py
"""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
panels: list[dict] = []
y = 0


def row(title: str) -> None:
    global y
    panels.append({"type": "row", "title": title, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
                   "collapsed": False, "panels": []})  # fmt: skip
    y += 1


def ts(title, targets, x, w=8, h=8, unit=None, log=False, thresholds=None, desc="", points=True):
    custom = {"drawStyle": "line", "lineWidth": 2, "showPoints": "always" if points else "never",
              "pointSize": 6, "spanNulls": False}  # fmt: skip
    defaults = {"custom": custom}
    if log:
        custom["scaleDistribution"] = {"type": "log", "log": 10}
        defaults["unit"] = "sci"
        defaults["decimals"] = 0
    if unit:
        defaults["unit"] = unit
    if thresholds:
        custom["thresholdsStyle"] = {"mode": "dashed"}
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
    panels.append({
        "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi"}},
        "targets": [{"datasource": DS, "expr": e, "legendFormat": lf, "refId": chr(65 + i)}
                    for i, (e, lf) in enumerate(targets)],
    })  # fmt: skip


def stat(title, expr, x, w=4, h=4, unit=None, legend="", thresholds=None, text=False):
    defaults = {"unit": unit} if unit else {}
    thresholds = thresholds or [{"color": "blue", "value": None}]
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
        defaults["color"] = {"mode": "thresholds"}
    panels.append({
        "type": "stat", "title": title, "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]},
                    "textMode": "name" if text else "value", "colorMode": "background"},
        "targets": [{"datasource": DS, "expr": expr, "legendFormat": legend, "refId": "A",
                     "instant": True}],
    })  # fmt: skip


GREEN, RED, AMBER = "green", "red", "orange"

row("Drift: reference window (validation months, champion) vs current window (prediction log)")
ts("Narrative length drift: -log10(KS p-value), alert above 3 (p < 0.001)",
   [('-log10(drift_feature_score{feature="narrative_len"})', "narrative_len")], 0, w=10,
   thresholds=[{"color": "transparent", "value": None}, {"color": RED, "value": 3}],
   desc="One point per drift job run (every 5 min, 10-min window, <= 300 rows). "
        "p = 0.1 plots as 1; p = 1e-23 as 23.")  # fmt: skip
ts("Drift, other columns: -log10(p-value)",
   [('-log10(drift_feature_score{feature=~"n_words|oov_share|mask_share|max_proba|predicted_queue"})',
     "{{feature}}")], 10, w=8)  # fmt: skip
stat("Alert threshold (p)", "vector(0.001)", 18, w=3, h=4)
stat("Window rows", "max(drift_window_rows)", 21, w=3, h=4)
y += 4
stat("Minutes since last window", "(time() - max(drift_run_timestamp_seconds)) / 60", 18, w=3,
     h=4, unit="none", thresholds=[{"color": GREEN, "value": None}, {"color": RED, "value": 15}])
stat("Champion (registry)", 'max by (model_version) (drift_model_info)', 21, w=3, h=4,
     legend="v{{model_version}}", text=True)  # fmt: skip
y += 4
ts("Share of drifted columns (Evidently, all 20 columns)", [("drift_share", "share")], 0, w=8,
   desc="Not alerted on: the company-mix columns drift between any two months (design 6.6).")
ts("Predicted queue mix: Jensen-Shannon distance to reference",
   [("prediction_queue_js_distance", "JS distance")], 8, w=8)
ts("Quality on closed complaints (labels joined to predictions)",
   [("quality_macro_f1", "macro-F1"), ("quality_accuracy", "accuracy")], 16, w=8)
y += 8
ts("Predicted queue share, current window",
   [('prediction_queue_share{window="current"}', "{{queue}}")], 0, w=12, unit="percentunit")
ts("Firing alerts", [('ALERTS{alertstate="firing"}', "{{alertname}}")], 12, w=12, points=False)
y += 8
row("Services: router (ML) and assistant (LLM)")
ts("Router requests / s by status",
   [('sum by (status) (rate(router_requests_total{endpoint="/predict"}[1m]))', "{{status}}")],
   0, w=8)
RQ = "sum by (le) (rate(router_request_latency_seconds_bucket[2m]))"
AQ = "sum by (le) (rate(assistant_request_latency_seconds_bucket[5m]))"
ts("Router latency", [(f"histogram_quantile(0.5, {RQ})", "p50"),
                      (f"histogram_quantile(0.95, {RQ})", "p95")], 8, w=8, unit="s",
   points=False)  # fmt: skip
ts("Router predictions / s by queue",
   [("sum by (queue) (rate(router_predictions_total[2m]))", "{{queue}}")], 16, w=8, points=False)
y += 8
stat("Router model", "max by (model_version, image_commit) (router_model_info)", 0, w=6,
     legend="v{{model_version}} image {{image_commit}}", text=True)
stat("Router ready", "max(router_ready)", 6, w=3,
     thresholds=[{"color": RED, "value": None}, {"color": GREEN, "value": 1}])
stat("Prediction log errors", "sum(router_prediction_log_errors_total)", 9, w=3,
     thresholds=[{"color": GREEN, "value": None}, {"color": RED, "value": 1}])
stat("Assistant ready", "max(assistant_ready)", 12, w=3,
     thresholds=[{"color": RED, "value": None}, {"color": GREEN, "value": 1}])
stat("Assistant model", "max by (llm_model, prompt_hash) (assistant_info)", 15, w=9,
     legend="{{llm_model}} prompt {{prompt_hash}}", text=True)
y += 4
ts("Assistant latency", [(f"histogram_quantile(0.5, {AQ})", "p50"),
                         (f"histogram_quantile(0.95, {AQ})", "p95")], 0, w=12, unit="s",
   points=False)  # fmt: skip
ts("Assistant requests / s and LLM tokens / s", [
    ('sum by (status) (rate(assistant_requests_total[5m]))', "requests {{status}}"),
    ('sum by (direction) (rate(assistant_llm_tokens_total[5m]))', "tokens {{direction}}"),
], 12, w=12, points=False)  # fmt: skip

dashboard = {
    "uid": "complaint-routing", "title": "Complaint routing", "editable": True,
    "schemaVersion": 39, "time": {"from": "now-1h", "to": "now"}, "refresh": "30s",
    "timezone": "browser", "tags": ["lab4"], "panels": panels,
}  # fmt: skip
out = Path(__file__).resolve().parents[1] / "k8s" / "monitoring" / "dashboard.json"
out.write_text(json.dumps(dashboard, indent=1) + "\n")
print(f"{out}: {len(panels)} panels")
