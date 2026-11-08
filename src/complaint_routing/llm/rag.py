"""The query path: question -> top-k chunks -> answer prompt (versioned file) -> LLM -> answer.

    uv run python -m complaint_routing.llm.rag "Can a collector call me at work?"
"""

import argparse
import re
from dataclasses import dataclass

from complaint_routing.llm import settings
from complaint_routing.llm.client import LLM, Call
from complaint_routing.llm.index import Hit, Retriever

CITATION = re.compile(r"ask-\d+")


@dataclass
class Answer:
    question: str
    hits: list[Hit]
    call: Call

    @property
    def cited(self) -> list[str]:
        return list(dict.fromkeys(CITATION.findall(self.call.text)))


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.doc_id}] {h.title}\n{h.text}" for h in hits)


class Assistant:
    def __init__(self, cfg: dict, retriever: Retriever | None = None, llm: LLM | None = None):
        self.cfg = cfg
        self.retriever = retriever or Retriever(cfg)
        self.llm = llm or LLM.from_config(cfg)

    def answer(self, question: str, hits: list[Hit] | None = None) -> Answer:
        k = self.cfg["retrieval"]["top_k"]
        hits = (hits or self.retriever.search(question, k))[:k]
        prompt = settings.render(
            self.cfg, "answer", question=question, context=format_context(hits)
        )
        return Answer(question, hits, self.llm.complete(prompt))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    cfg = settings.load(overrides=args.set)
    a = Assistant(cfg).answer(args.question)
    for h in a.hits:
        print(f"  {h.distance:.3f}  {h.chunk_id:<14} {h.title}")
    print(f"\n{a.call.text}\n")
    print(
        f"[{a.call.prompt_tokens} tokens in, {a.call.completion_tokens} out,"
        f" {a.call.latency_s:.1f} s, prompt {settings.prompt_hash(cfg, 'answer')}]"
    )


if __name__ == "__main__":
    main()
