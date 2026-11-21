"""Smoke request against the prediction endpoint: one complaint in, a queue and a version out.

    python -m complaint_routing.serving.smoke http://router        # inside the cluster
    uv run python -m complaint_routing.serving.smoke http://localhost:8081
"""

import json
import sys
import time
import urllib.request

COMPLAINT = {
    "narrative": "I paid off my car loan in full in March but the lender keeps reporting the "
    "account as late and charging me fees. I sent the payoff letter twice and nobody answers.",
    "company": "none",
    "state": "TX",
    "submitted_via": "Web",
    "tags": "none",
}


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8081").rstrip("/")
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"{base}/health/ready", timeout=5) as r:
                ready = json.load(r)
            break
        except Exception as e:  # not ready yet: 503 or connection refused
            print(f"waiting for {base}/health/ready ({e})", flush=True)
            time.sleep(5)
    else:
        sys.exit(f"{base} never became ready")
    req = urllib.request.Request(
        f"{base}/predict", json.dumps(COMPLAINT).encode(), {"Content-Type": "application/json"}
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as r:
        out = json.load(r)
    top = sorted(out["probabilities"].items(), key=lambda kv: -kv[1])[:3]
    print(f"ready: {ready['ready']}, image {ready['image_commit']}")
    print(
        f"POST {base}/predict -> queue {out['queue']} (top 3 {top}),"
        f" {out['model_name']}@{out['model_alias']} = v{out['model_version']},"
        f" {1000 * (time.perf_counter() - t):.0f} ms"
    )


if __name__ == "__main__":
    main()
