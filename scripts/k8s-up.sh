#!/usr/bin/env bash
# One command brings the system up: `make k8s-up`.
# Storage next to the cluster (docker compose) -> preflight (registry, reference, features,
# index) -> Minikube -> images tagged with the commit, built inside Minikube -> Secret and
# ConfigMaps -> every workload -> wait for readiness -> a smoke request through the Service.
# Idempotent: a second run reuses the cluster and the images of the same commit.
set -euo pipefail
cd "$(dirname "$0")/.."
START=$(date +%s)
PROFILE=${MINIKUBE_PROFILE:-complaint-routing}
NS=complaint-routing
K="kubectl --context $PROFILE -n $NS"
step() { printf '\n[%4ss] == %s\n' "$(($(date +%s) - START))" "$*"; }

step "0/8 local environment (.env, uv)"
[ -f .env ] || cp .env.example .env
set -a; . ./.env; set +a
export WAREHOUSE_DSN=${WAREHOUSE_DSN:-postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:5433/warehouse}
uv sync --frozen -q
COMMIT=$(git rev-parse HEAD)
TAG=$(git rev-parse --short=12 HEAD)
if [ -n "$(git status --porcelain)" ]; then
  TAG="$TAG-dirty"   # an image built from uncommitted code never carries a bare commit tag
  echo "working tree is dirty: images are tagged $TAG"
fi
echo "image tag: $TAG"

step "1/8 storage next to the cluster: PostgreSQL, MinIO, MLflow (docker compose); vLLM"
docker compose up -d postgres minio minio-init mlflow
if docker image inspect vllm-cpu-arm64:0.30.0 >/dev/null 2>&1; then
  docker compose --profile llm up -d vllm
else
  echo "WARNING: image vllm-cpu-arm64:0.30.0 not built (make llm-image, Lab 3): the assistant stays not ready"
fi
until curl -sf localhost:5050/health >/dev/null; do sleep 2; done

step "2/8 preflight: registry, reference window, features, retrieval index"
uv run python scripts/preflight.py

step "3/8 Minikube ($PROFILE: Kubernetes v1.30.0, docker driver, 4 GB, 2 CPUs)"
if ! minikube -p "$PROFILE" status >/dev/null 2>&1; then
  minikube start -p "$PROFILE" --driver=docker --memory 4096 --cpus 2 --kubernetes-version=v1.30.0
fi
# How pods reach the storage that runs next to the cluster (ports published on the host).
COMPOSE_HOST=${COMPOSE_HOST:-host.minikube.internal}

step "4/8 images, tag = commit $TAG, built inside Minikube"
have() { minikube -p "$PROFILE" image ls 2>/dev/null | grep -q "$1"; }
have "complaint-routing/mlflow:3.16.1" ||
  minikube -p "$PROFILE" image build -t complaint-routing/mlflow:3.16.1 docker/mlflow
for target in router assistant monitor; do
  if have "complaint-routing/$target:$TAG"; then
    echo "complaint-routing/$target:$TAG already built"
  else
    minikube -p "$PROFILE" image build -t "complaint-routing/$target:$TAG" \
      -f docker/service/Dockerfile --build-opt="target=$target" \
      --build-opt="build-arg=GIT_COMMIT=$COMMIT" .
  fi
done

step "5/8 namespace, Secret (from .env), ConfigMaps"
kubectl --context "$PROFILE" apply -f k8s/00-namespace.yaml
PG="$POSTGRES_USER:$POSTGRES_PASSWORD@$COMPOSE_HOST:5433"
$K create secret generic complaint-routing-secrets --dry-run=client -o yaml \
  --from-literal=WAREHOUSE_DSN="postgresql://$PG/warehouse" \
  --from-literal=RAG_DB_URL="postgresql://$PG/warehouse" \
  --from-literal=MLFLOW_BACKEND_URI="postgresql+psycopg2://$PG/mlflow" \
  --from-literal=AWS_ACCESS_KEY_ID="$MINIO_ROOT_USER" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$MINIO_ROOT_PASSWORD" \
  --from-literal=LLM_API_KEY="${LLM_API_KEY:-unused}" | $K apply -f -
$K create configmap llm-config --from-file=llm.yaml=config/llm.yaml --dry-run=client -o yaml | $K apply -f -
$K create configmap prometheus-config --dry-run=client -o yaml \
  --from-file=k8s/monitoring/prometheus.yml --from-file=k8s/monitoring/rules.yml | $K apply -f -
$K create configmap grafana-provisioning --dry-run=client -o yaml \
  --from-file=datasources.yml=k8s/monitoring/grafana-datasources.yml \
  --from-file=dashboards.yml=k8s/monitoring/grafana-dashboards.yml | $K apply -f -
$K create configmap grafana-dashboards --dry-run=client -o yaml \
  --from-file=complaint-routing.json=k8s/monitoring/dashboard.json | $K apply -f -
AM=$(mktemp)
trap 'rm -f "$AM"' EXIT
uv run python scripts/render_alertmanager.py > "$AM"
$K create secret generic alertmanager-config --from-file=alertmanager.yml="$AM" \
  --dry-run=client -o yaml | $K apply -f -

step "6/8 workloads: MLflow, router, assistant, monitoring, drift CronJob"
hash_of() { cat "$@" | shasum | cut -c1-12; }
render() {
  sed -e "s|__IMAGE_TAG__|$TAG|g" -e "s|__COMPOSE_HOST__|$COMPOSE_HOST|g" \
    -e "s|__LLM_CONFIG_HASH__|$(hash_of config/llm.yaml k8s/10-config.yaml)|g" \
    -e "s|__PROMETHEUS_CONFIG_HASH__|$(hash_of k8s/monitoring/prometheus.yml k8s/monitoring/rules.yml)|g" \
    -e "s|__ALERTMANAGER_CONFIG_HASH__|$(hash_of "$AM")|g" \
    -e "s|__GRAFANA_CONFIG_HASH__|$(hash_of k8s/monitoring/dashboard.json k8s/monitoring/grafana-*.yml)|g" \
    "$1"
}
for f in k8s/[1-9]*.yaml; do render "$f" | $K apply -f -; done

step "7/8 waiting for every Deployment to be ready"
for d in mlflow router prometheus pushgateway alertmanager grafana; do
  $K rollout status "deployment/$d" --timeout=600s
done
if ! $K rollout status deployment/assistant --timeout=300s; then
  echo "WARNING: the assistant is not ready; why:"
  $K exec deploy/assistant -- python -c "import urllib.request as u; print(u.urlopen('http://localhost:8000/health/ready').read())" || true
fi

step "8/8 smoke request: POST http://router/predict from inside the cluster"
$K delete pod smoke --ignore-not-found >/dev/null
$K run smoke --rm -i --restart=Never --quiet --image="complaint-routing/router:$TAG" \
  --image-pull-policy=Never -- python -m complaint_routing.serving.smoke http://router

$K get pods,svc,cronjob
step "done in $(($(date +%s) - START)) s. Host access: make k8s-forward"
