"""Retrieval metrics computed by our own code (no evaluation library), at document level.

Relevance is labelled per document, not per chunk: chunk ids change with the chunk size, and
the chunk-size sweep must be scored against the same labels. A retrieved chunk is relevant if
its document is one of the question's relevant documents.
"""

from collections.abc import Sequence


def first_relevant_rank(retrieved_docs: Sequence[str], relevant: set[str]) -> int | None:
    """1-based rank of the first retrieved chunk whose document is relevant, or None."""
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc in relevant:
            return rank
    return None


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mrr(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks)
