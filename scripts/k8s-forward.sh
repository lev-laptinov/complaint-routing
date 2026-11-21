#!/usr/bin/env bash
# Host access to the cluster's Services on 127.0.0.1 (kubectl port-forward, restarted if a pod
# is replaced). `make k8s-forward` starts them in the background; `make k8s-forward-stop` ends them.
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
       "svc/$svc" "$local:$remote" >/dev/null 2>&1 || true
     sleep 2
   done) &
  echo $! >>"$PIDS"
  printf '%-13s http://localhost:%s\n' "$svc" "$local"
done
