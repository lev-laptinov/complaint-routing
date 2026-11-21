"""Writes alertmanager.yml (to stdout) from .env: e-mail to a human, plus the retraining webhook.

    uv run python scripts/render_alertmanager.py > alertmanager.yml

E-mail needs ALERT_SMTP_USER, ALERT_SMTP_PASSWORD and ALERT_EMAIL_TO in .env (for Gmail, an app
password); ALERT_SMTP_HOST defaults to smtp.gmail.com:587. Without them the alerts still reach
Alertmanager's UI and Grafana's "Firing alerts" panel, and a warning says that no e-mail goes out.
The output contains the SMTP password, so the bring-up stores it as a Secret, never in git.
"""

import os
import sys

import yaml

user, password, to = (os.environ.get(k, "") for k in
                      ("ALERT_SMTP_USER", "ALERT_SMTP_PASSWORD", "ALERT_EMAIL_TO"))  # fmt: skip
human: dict = {"name": "human"}
if user and password and to:
    human["email_configs"] = [{
        "to": to,
        "from": f"Complaint routing alerts <{user}>",
        "smarthost": os.environ.get("ALERT_SMTP_HOST", "smtp.gmail.com:587"),
        "auth_username": user,
        "auth_password": password,
        "require_tls": True,
        "send_resolved": True,
        "headers": {"Subject": "[complaint-routing] {{ .Status | toUpper }}: "
                               "{{ .CommonLabels.alertname }} ({{ .CommonLabels.severity }})"},
    }]  # fmt: skip
else:
    print("WARNING: ALERT_SMTP_USER / ALERT_SMTP_PASSWORD / ALERT_EMAIL_TO not set in .env:"
          " alerts are visible in Alertmanager and Grafana only, no e-mail", file=sys.stderr)

config = {
    "global": {"resolve_timeout": "5m"},
    "route": {
        "receiver": "human",
        "group_by": ["alertname"],
        "group_wait": "30s",
        "group_interval": "5m",
        "repeat_interval": "4h",
        "routes": [
            # Stretch goal: a drift alert also starts a retraining run (k8s/60-retrain.yaml).
            {"matchers": ['action="retrain"'], "receiver": "retrain", "continue": True},
            {"receiver": "human"},
        ],
    },
    "receivers": [
        human,
        {"name": "retrain", "webhook_configs": [
            {"url": "http://retrain-trigger:8080/alert", "send_resolved": False}]},
    ],
}  # fmt: skip
yaml.safe_dump(config, sys.stdout, sort_keys=False)
