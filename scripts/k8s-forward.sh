#!/usr/bin/env bash
# Host access to the cluster's Services on 127.0.0.1 (kubectl port-forward). A port-forward to a
# Service sticks to one pod and hangs when that pod is replaced (rollout, promotion), so each one
# has a watchdog: an HTTP check every 5 s, and a restart when it fails.
# `make k8s-forward` starts them in the background; `make k8s-forward-stop` ends them.
set -euo pipefail
PROFILE=${MINIKUBE_PROFILE:-complaint-routing}
PIDS=${TMPDIR:-/tmp}/complaint-routing-forward.pids
if [ "${1:-}" = stop ]; then
  [ -f "$PIDS" ] && xargs kill <"$PIDS" 2>/dev/null || true
  pkill -f "kubectl --context $PROFILE -n complaint-routing port-forward" 2>/dev/null || true
  rm -f "$PIDS"; echo "port-forwards stopped"; exit 0
fi
"$0" stop >/dev/null
for pair in router:8081:80 assistant:8082:80 grafana:3000:3000 prometheus:9090:9090 \
            alertmanager:9093:9093 pushgateway:9091:9091 mlflow:5051:5000; do
  IFS=: read -r svc local remote <<<"$pair"
  (while true; do
     kubectl --context "$PROFILE" -n complaint-routing port-forward --address 127.0.0.1 \
       "svc/$svc" "$local:$remote" >/dev/null 2>&1 &
     pf=$!
     sleep 3
     while kill -0 "$pf" 2>/dev/null && curl -s -o /dev/null -m 5 "http://127.0.0.1:$local/"; do
       sleep 5
     done
     kill "$pf" 2>/dev/null || true
     sleep 1
   done) &
  echo $! >>"$PIDS"
  printf '%-13s http://localhost:%s\n' "$svc" "$local"
done
