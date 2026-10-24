"""Reproduce a registered model version from its two lineage tags, with one command.

    uv run python -m complaint_routing.reproduce --model complaint-router --alias champion

1. Read data_version and git_commit from the registered version, and its params and metric.
2. Check out that commit in a throwaway git worktree (the code exactly as it was).
3. `dvc pull` there: the .dvc file of that commit fetches that data version from the remote.
4. Train with the logged params in the worktree's own environment (built from its uv.lock).
5. Compare test macro-F1 with the registered run; exit 1 if it differs by more than --tolerance.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from mlflow.tracking import MlflowClient

from complaint_routing import lineage
from complaint_routing.config import ROOT


def run(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="complaint-router")
    p.add_argument("--alias", default="champion")
    p.add_argument("--version", default=None, help="model version; overrides --alias")
    p.add_argument("--metric", default="test_macro_f1")
    p.add_argument("--tolerance", type=float, default=0.005)
    args = p.parse_args()

    client = MlflowClient()
    if args.version:
        mv = client.get_model_version(args.model, args.version)
    else:
        mv = client.get_model_version_by_alias(args.model, args.alias)
    original = client.get_run(mv.run_id)
    commit, data = mv.tags["git_commit"], mv.tags["data_version"]
    params, target = original.data.params, original.data.metrics[args.metric]
    plan = {"model": args.model, "version": mv.version, "run_id": mv.run_id,
            "git_commit": commit, "data_version": data, args.metric: target}  # fmt: skip
    print(json.dumps(plan, indent=1))

    workdir = Path(tempfile.mkdtemp(prefix="repro-"))
    result = Path(tempfile.mkdtemp(prefix="repro-result-")) / "result.json"
    run(["git", "worktree", "add", "--detach", str(workdir), commit], cwd=ROOT)
    try:
        # Where the remote lives is infrastructure, not lineage: the data hash comes from the
        # commit, the remote address from today's config (it moved since some commits were made).
        endpoint = subprocess.check_output(
            ["uv", "run", "dvc", "config", "remote.minio.endpointurl"], cwd=ROOT, text=True
        ).strip()
        run(["uv", "run", "--frozen", "dvc", "remote", "modify", "--local", "minio",
             "endpointurl", endpoint], cwd=workdir)  # fmt: skip
        run(["uv", "run", "--frozen", "dvc", "pull", "data/raw.dvc"], cwd=workdir)
        pulled = lineage.data_version(workdir / "data" / "raw.dvc")
        if pulled != data:
            sys.exit(f"data version mismatch: commit has {pulled}, model was trained on {data}")
        train = ["uv", "run", "--frozen", "python", "-m", "complaint_routing.train",
                 "--no-register", "--only-candidate",
                 "--experiment", "complaint-routing-reproductions",
                 "--reproduces", mv.run_id, "--result-file", str(result),
                 "--c", params["C"], "--max-features", params["max_features"],
                 "--min-df", params["min_df"], "--seed", params["seed"]]  # fmt: skip
        run(train, cwd=workdir)
        new = json.loads(result.read_text())
        value = new["metrics"][args.metric]
    finally:
        run(["git", "worktree", "remove", "--force", str(workdir)], cwd=ROOT)

    diff = abs(value - target)
    ok = diff <= args.tolerance
    verdict = "REPRODUCED" if ok else "NOT REPRODUCED"
    print(f"\n{args.metric}: registered {target:.4f}, reproduced {value:.4f} (run {new['run_id']})")
    print(f"|diff| {diff:.4f} {'<=' if ok else '>'} tolerance {args.tolerance} -> {verdict}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
