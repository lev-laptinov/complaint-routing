"""LLM baseline for the routing task itself: zero-shot queue prediction, logged next to Lab 2.

    make llm-baseline     # = uv run python -m complaint_routing.llm.classify

The LLM and the registered champion are scored on the same seeded sample of the test months
(2026-05 -> 07) with the same metric code as Lab 2 (train.scores), and both runs go to the Lab 2
experiment `complaint-routing`, so the two baselines sit side by side in the MLflow UI.
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from complaint_routing import config
from complaint_routing.llm import settings
from complaint_routing.llm.client import LLM, usage_metrics
from complaint_routing.train import MODEL_NAME, lineage_tags, load_dataset, scores, split

UNPARSED = "unparsed"


def parse_queue(text: str, queues: list[str]) -> str:
    """The first queue name in the reply (case, spaces and hyphens tolerated), else 'unparsed'."""
    norm = re.sub(r"[\s-]+", "_", text.lower())
    found = [(norm.find(q), q) for q in queues if q in norm]
    return min(found)[1] if found else UNPARSED


def llm_params(cfg: dict) -> dict:
    return {
        "llm_model": cfg["llm"]["model"],
        "llm_base_url": cfg["llm"]["base_url"],
        "temperature": cfg["llm"]["temperature"],
        "max_tokens": cfg["classify"]["max_tokens"],
        "max_narrative_chars": cfg["classify"]["max_narrative_chars"],
        "shots": 0,
        "retrieval": "off",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--experiment", default="complaint-routing")
    args = p.parse_args()
    cfg = settings.load(overrides=args.set)
    c = cfg["classify"]
    queues = config.queues()

    test = split(load_dataset())["test"]
    sample = test.sample(n=c["sample_size"], random_state=c["seed"]).sort_values("complaint_id")
    print(f"{len(sample)} of {len(test)} test complaints; queue mix:")
    print(sample["queue"].value_counts().to_string())

    llm = LLM.from_config(cfg)
    rows = []
    for n, row in enumerate(sample.to_dict("records"), 1):
        prompt = settings.render(
            cfg, "classify", narrative=row["narrative"][: c["max_narrative_chars"]]
        )
        call = llm.complete(prompt, max_tokens=c["max_tokens"])
        rows.append({"complaint_id": row["complaint_id"], "queue": row["queue"], "reply": call.text,
                     "pred": parse_queue(call.text, queues), "latency_s": call.latency_s,
                     "prompt_tokens": call.prompt_tokens})  # fmt: skip
        if n % 25 == 0:
            print(f"{n}/{len(sample)}  last: {row['queue']} -> {call.text!r}")
    res = pd.DataFrame(rows)

    mlflow.set_experiment(args.experiment)
    tags = lineage_tags() | {
        "model_type": "llm",
        "eval_set": f"test-sample-{c['sample_size']}-seed{c['seed']}",
        "prompt_hash": settings.prompt_hash(cfg, "classify"),
        "config_hash": settings.config_hash(cfg),
    }
    common = {"test_months": "..".join(config.TEST_MONTHS), "n_test": len(sample),
              "sample_seed": c["seed"]}  # fmt: skip
    metrics = scores(res["queue"], res["pred"], queues, "test")
    metrics |= {"test_unparsed_rate": float((res["pred"] == UNPARSED).mean())}
    metrics |= usage_metrics(llm.calls, "llm")
    name = f"llm-zero-shot-{cfg['llm']['model'].split('/')[-1]}"
    with mlflow.start_run(run_name=name, tags=tags) as run:
        prompt = {"prompt_file": cfg["prompts"]["classify"], "prompt_hash": tags["prompt_hash"]}
        mlflow.log_params(common | llm_params(cfg) | prompt)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(settings.prompt_path(cfg, "classify")), "prompt")
        mlflow.log_artifact(str(settings.CONFIG_PATH), "config")
        with tempfile.TemporaryDirectory() as d:
            res.to_csv(Path(d) / "predictions.csv", index=False)
            mlflow.log_artifact(str(Path(d) / "predictions.csv"), "evaluation")
        llm_run = run.info.run_id

    # The champion on exactly the same rows, so the comparison is not test-set-vs-sample.
    champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    champ_metrics = scores(sample["queue"], champion.predict(sample), queues, "test")
    with mlflow.start_run(
        run_name="champion-on-llm-sample", tags=tags | {"model_type": "trained"}
    ) as run:
        mlflow.log_params(common | {"model": f"models:/{MODEL_NAME}@champion"})
        mlflow.log_metrics(champ_metrics)

    keys = ["test_macro_f1", "test_accuracy", "test_min_recall"]
    print(json.dumps({
        "llm_run": llm_run, "llm": {k: round(metrics[k], 4) for k in keys},
        "champion_same_rows": {k: round(champ_metrics[k], 4) for k in keys},
        "unparsed_rate": metrics["test_unparsed_rate"],
        "latency_p50_s": round(metrics["llm_latency_p50_s"], 2),
        "latency_p95_s": round(metrics["llm_latency_p95_s"], 2),
    }, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
