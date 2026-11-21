"""The model registry as the deployment interface: aliases, not files or version numbers.

    uv run python -m complaint_routing.registry show
    uv run python -m complaint_routing.registry promote --version 2   # champion -> v2
    uv run python -m complaint_routing.registry promote --version 1   # rollback

Promotion moves the alias `champion` to a version and keeps the version it replaced under
`previous`, so a rollback is the same command with that number. Nothing is rebuilt: the routing
service resolves the alias when it starts (`kubectl rollout restart deployment/router`).
"""

import argparse
from dataclasses import dataclass

from mlflow.tracking import MlflowClient

MODEL_NAME = "complaint-router"


@dataclass
class Resolved:
    name: str
    alias: str
    version: str
    run_id: str
    tags: dict[str, str]

    @property
    def uri(self) -> str:
        # Load the exact version the alias pointed to when it was resolved, so the version the
        # service reports is the one it loaded even if the alias moves during the download.
        return f"models:/{self.name}/{self.version}"


def resolve(name: str, alias: str, client: MlflowClient | None = None) -> Resolved:
    mv = (client or MlflowClient()).get_model_version_by_alias(name, alias)
    return Resolved(name, alias, str(mv.version), mv.run_id, dict(mv.tags))


def promote(name: str, version: str, alias: str = "champion") -> None:
    client = MlflowClient()
    try:
        current = resolve(name, alias, client).version
    except Exception:
        current = None
    if current == str(version):
        print(f"{name}@{alias} is already v{version}")
        return
    client.get_model_version(name, str(version))  # fails loudly on a version that does not exist
    if current is not None:
        client.set_registered_model_alias(name, "previous", current)
    client.set_registered_model_alias(name, alias, str(version))
    kept = f" (previous = v{current})" if current else ""
    print(f"{name}@{alias}: v{current} -> v{version}{kept}")


def show(name: str) -> None:
    client = MlflowClient()
    found = client.search_model_versions(f"name='{name}'")
    for v in sorted((m.version for m in found), key=int):
        mv = client.get_model_version(name, v)  # search results carry no aliases
        aliases = ",".join(mv.aliases) or "-"
        tags = {k: mv.tags.get(k) for k in ("git_commit", "data_version", "trigger")}
        print(f"v{mv.version:<3} aliases={aliases:<20} run={mv.run_id[:8]} {tags}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    pr = sub.add_parser("promote")
    pr.add_argument("--version", required=True)
    pr.add_argument("--alias", default="champion")
    p.add_argument("--model", default=MODEL_NAME)
    args = p.parse_args()
    if args.cmd == "show":
        show(args.model)
    else:
        promote(args.model, args.version, args.alias)


if __name__ == "__main__":
    main()
