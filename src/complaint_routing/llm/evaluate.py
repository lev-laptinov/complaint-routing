"""Evaluate retrieval and answers on the fixed question set; every evaluation is an MLflow run.

    make rag-eval                                   # the configuration in config/llm.yaml
    make rag-sweep PARAM=retrieval.chunk_size VALUES=60,120,240
    uv run python -m complaint_routing.llm.evaluate --retrieval-only --sweep retrieval.top_k=1,3,5

Retrieval: hit rate@k and MRR@k at document level (metrics.py), from the top `max_k` chunks.
Answers: the answer prompt over the top `top_k` chunks, graded YES/NO against the reference
answer by an LLM judge whose prompt is a versioned, hashed file like every other prompt.
A sweep changes one setting at a time, builds any index it needs (under its own index version,
so the served index is untouched) and prints the comparison table.
"""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import mlflow

from complaint_routing import lineage
from complaint_routing.config import ROOT
from complaint_routing.llm import index, settings
from complaint_routing.llm.client import LLM, usage_metrics
from complaint_routing.llm.metrics import first_relevant_rank, hit_rate, mrr
from complaint_routing.llm.rag import Assistant

EXPERIMENT = "complaint-assistant-rag"
REFUSAL = "do not answer"


def load_questions(cfg: dict) -> list[dict]:
    with open(ROOT / cfg["evaluation"]["questions"]) as f:
        return [json.loads(line) for line in f if line.strip()]


def answer_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_hand_grades(cfg: dict) -> dict[tuple[str, str], int]:
    path = ROOT / cfg["evaluation"]["hand_grades"]
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(r["qid"], r["answer_sha"]): int(r["hand_correct"]) for r in csv.DictReader(f)}


def hand_metrics(rows: list[dict], grades: dict[tuple[str, str], int]) -> dict[str, float]:
    """Hand rubric on the graded sample, and how often the LLM judge agrees with it."""
    graded = [(grades[k], r["correct"]) for r in rows
              if (k := (r["qid"], answer_sha(r["answer"]))) in grades]  # fmt: skip
    if not graded:
        return {}
    return {
        "hand_graded_n": len(graded),
        "answer_correct_hand": sum(h for h, _ in graded) / len(graded),
        "judge_agreement_hand": sum(h == j for h, j in graded) / len(graded),
        "judge_false_yes_hand": sum(h == 0 and j == 1 for h, j in graded) / len(graded),
    }


def judge_verdict(text: str) -> int | None:
    word = text.strip().split()[0].strip(".,:!").upper() if text.strip() else ""
    return {"YES": 1, "NO": 0}.get(word)


def ensure_index(cfg: dict) -> dict:
    info = index.describe(cfg)
    if info is None:
        print(f"building index {settings.index_version(cfg)} for this setting")
        index.build(cfg)
        info = index.describe(cfg)
    return info


def evaluate(cfg: dict, retrieval_only: bool, run_name: str | None, overrides: list[str]) -> dict:
    questions = load_questions(cfg)
    info = ensure_index(cfg)
    retriever = index.Retriever(cfg)
    max_k, top_k = cfg["evaluation"]["max_k"], cfg["retrieval"]["top_k"]
    assistant = None if retrieval_only else Assistant(cfg, retriever=retriever)
    judge = None if retrieval_only else LLM.from_config(cfg, model=cfg["judge"]["model"])

    rows = []
    for n, q in enumerate(questions, 1):
        hits = retriever.search(q["question"], max_k)
        rank = first_relevant_rank([h.doc_id for h in hits], set(q["relevant_docs"]))
        row = {"qid": q["qid"], "question": q["question"], "relevant_docs": q["relevant_docs"],
               "retrieved": [h.chunk_id for h in hits], "rank": rank}  # fmt: skip
        if assistant:
            ans = assistant.answer(q["question"], hits)
            retrieved_docs = {h.doc_id for h in ans.hits}
            verdict = judge.complete(
                settings.render(cfg, "judge", question=q["question"],
                                reference=q["reference_answer"], answer=ans.call.text),
                max_tokens=cfg["judge"]["max_tokens"],
            )  # fmt: skip
            row |= {
                "answer": ans.call.text,
                "cited": ans.cited,
                "cited_outside_retrieved": [d for d in ans.cited if d not in retrieved_docs],
                "refused": REFUSAL in ans.call.text.lower(),
                "judge_reply": verdict.text,
                "correct": judge_verdict(verdict.text),
                "latency_s": ans.call.latency_s,
            }
            print(f"{n:>3}/{len(questions)} rank={rank} judge={verdict.text!r:6} "
                  f"{ans.call.latency_s:5.1f}s  {q['question'][:60]}")  # fmt: skip
        rows.append(row)

    ranks = [r["rank"] for r in rows]
    metrics = {f"hit_rate_at_{k}": hit_rate(ranks, k) for k in (1, 3, 5, 10) if k <= max_k}
    metrics |= {f"mrr_at_{max_k}": mrr(ranks, max_k), f"mrr_at_{top_k}": mrr(ranks, top_k),
                "hit_rate_at_top_k": hit_rate(ranks, top_k)}  # fmt: skip
    if assistant:
        graded = [r["correct"] for r in rows if r["correct"] is not None]
        metrics |= {
            "answer_correct": sum(graded) / len(rows),  # an unparseable verdict counts as wrong
            "judge_unparsed_rate": 1 - len(graded) / len(rows),
            "refusal_rate": sum(r["refused"] for r in rows) / len(rows),
            "citation_rate": sum(bool(r["cited"]) for r in rows) / len(rows),
            "cited_outside_retrieved_rate":
                sum(bool(r["cited_outside_retrieved"]) for r in rows) / len(rows),
        }  # fmt: skip
        metrics |= hand_metrics(rows, load_hand_grades(cfg))
        metrics |= usage_metrics(assistant.llm.calls, "llm")
        metrics |= usage_metrics(judge.calls, "judge")

    r, c = cfg["retrieval"], cfg["llm"]
    params = {
        "chunk_size": r["chunk_size"], "chunk_overlap": r["chunk_overlap"],
        "embedding_model": r["embedding_model"], "top_k": top_k, "max_k": max_k,
        "corpus": r["corpus"], "corpus_version": settings.corpus_version(cfg),
        "index_version": info["index_version"], "n_documents": info["documents"],
        "n_chunks": info["chunks"], "eval_set": cfg["evaluation"]["questions"],
        "eval_set_md5": lineage.file_md5(ROOT / cfg["evaluation"]["questions"]),
        "hand_grades": cfg["evaluation"]["hand_grades"],
        "n_questions": len(rows), "retrieval_only": retrieval_only,
        "overrides": ";".join(overrides) or "none", "config_hash": settings.config_hash(cfg),
    }  # fmt: skip
    prompts = {f"prompt_hash_{p}": settings.prompt_hash(cfg, p) for p in ("answer", "judge")}
    if not retrieval_only:
        params |= {"llm_model": c["model"], "llm_base_url": c["base_url"],
                   "temperature": c["temperature"], "max_tokens": c["max_tokens"],
                   "judge_model": cfg["judge"]["model"] or c["model"]}  # fmt: skip
        params |= prompts
    tags = {
        "git_commit": lineage.git_commit(),
        "git_dirty": str(lineage.git_dirty()).lower(),
        "config_hash": params["config_hash"],
        "index_version": info["index_version"],
        "corpus_version": params["corpus_version"],
    } | ({} if retrieval_only else prompts)
    mlflow.set_experiment(EXPERIMENT)

    def short(o: str) -> str:  # "retrieval.embedding_model=BAAI/x" -> "embedding_model=x"
        key, _, value = o.partition("=")
        return f"{key.split('.')[-1]}={value.split('/')[-1]}"

    name = run_name or ("retrieval" if retrieval_only else "rag") + (
        "-" + "-".join(short(o) for o in overrides) if overrides else ""
    )
    with mlflow.start_run(run_name=name, tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(settings.CONFIG_PATH), "config")
        if not retrieval_only:
            for p in ("answer", "judge"):
                mlflow.log_artifact(str(settings.prompt_path(cfg, p)), "prompts")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "per_question.jsonl"
            out.write_text("".join(json.dumps(row) + "\n" for row in rows))
            mlflow.log_artifact(str(out), "evaluation")
        run_id = run.info.run_id
    return {"run_id": run_id, "params": params, "metrics": metrics, "rows": rows}


def table(results: list[tuple[str, dict]], param: str) -> str:
    keys = ["hit_rate_at_1", "hit_rate_at_5", "hit_rate_at_10", "mrr_at_10",
            "answer_correct_hand", "hand_graded_n", "answer_correct", "judge_agreement_hand",
            "llm_latency_p50_s", "llm_latency_p95_s", "llm_tokens_in_per_call"]  # fmt: skip
    present = [k for k in keys if any(k in res["metrics"] for _, res in results)]
    head = f"| {param} | n_chunks | " + " | ".join(present) + " | run |"
    lines = [head, "|" + "---|" * (len(present) + 3)]
    for value, res in results:
        m = res["metrics"]
        cells = [f"{m[k]:.3f}" if k in m else "—" for k in present]
        lines.append(f"| {value} | {res['params']['n_chunks']} | " + " | ".join(cells)
                     + f" | `{res['run_id'][:8]}` |")  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--sweep", default=None, metavar="KEY=V1,V2,V3")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    if not args.sweep:
        cfg = settings.load(overrides=args.set)
        res = evaluate(cfg, args.retrieval_only, args.run_name, args.set)
        rounded = {k: round(v, 4) for k, v in res["metrics"].items()}
        print(json.dumps({"run_id": res["run_id"]} | rounded, indent=1))
        return
    key, _, values = args.sweep.partition("=")
    results = []
    for value in values.split(","):
        overrides = args.set + [f"{key}={value}"]
        cfg = settings.load(overrides=overrides)
        results.append((value, evaluate(cfg, args.retrieval_only, None, overrides)))
    print("\n" + table(results, key.split(".")[-1]))


if __name__ == "__main__":
    main()
