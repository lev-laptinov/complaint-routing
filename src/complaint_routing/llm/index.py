"""Ingestion (corpus -> chunks -> embeddings -> pgvector) and the retrieval half of the query path.

    make rag-index      # = uv run python -m complaint_routing.llm.index

Rebuilds the index for the settings in config/llm.yaml and prints the document and chunk counts.
Rows are keyed by the index version (settings.index_version), so indexes for different chunk
sizes or embedding models live side by side in rag.chunks and an experiment never overwrites
the index that is served. A rebuild of the same version deletes and re-inserts it in one
transaction, like the warehouse partitions of Lab 2.

Search is exact cosine distance (`<=>`) over the rows of one version: a few thousand chunks
need no approximate (HNSW) index, and exact search keeps the evaluation free of index recall.
"""

import argparse
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from complaint_routing.llm import settings
from complaint_routing.llm.corpus import load_corpus

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE TABLE IF NOT EXISTS rag.indexes (
    index_version text PRIMARY KEY,
    corpus_version text NOT NULL,
    chunk_size int NOT NULL,
    chunk_overlap int NOT NULL,
    embedding_model text NOT NULL,
    n_documents int NOT NULL,
    n_chunks int NOT NULL,
    built_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS rag.chunks (
    index_version text NOT NULL REFERENCES rag.indexes ON DELETE CASCADE,
    chunk_id text NOT NULL,
    doc_id text NOT NULL,
    title text NOT NULL,
    url text NOT NULL,
    text text NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (index_version, chunk_id)
);
"""


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window over words: `size` words per chunk, `overlap` shared with the next one."""
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    starts = range(0, len(words) - overlap, step)
    return [" ".join(words[s : s + size]) for s in starts]


def chunk_corpus(docs: list[dict], size: int, overlap: int) -> list[dict]:
    return [
        {"chunk_id": f"{d['doc_id']}#{n}", "doc_id": d["doc_id"], "title": d["title"],
         "url": d["url"], "text": text}
        for d in docs
        for n, text in enumerate(chunk_words(d["text"], size, overlap))
    ]  # fmt: skip


@lru_cache(maxsize=4)
def embedder(model_name: str):
    from sentence_transformers import SentenceTransformer  # heavy import, only when embedding

    return SentenceTransformer(model_name, device="cpu")


def embed(model_name: str, texts: list[str]) -> np.ndarray:
    return embedder(model_name).encode(
        texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
    )


def connect(cfg: dict) -> psycopg.Connection:
    conn = psycopg.connect(settings.db_url(cfg))
    conn.execute(DDL)
    conn.commit()
    register_vector(conn)
    return conn


def build(cfg: dict) -> dict:
    r = cfg["retrieval"]
    docs = load_corpus(settings.corpus_path(cfg))
    chunks = chunk_corpus(docs, r["chunk_size"], r["chunk_overlap"])
    vectors = embed(r["embedding_model"], [c["text"] for c in chunks])
    version = settings.index_version(cfg)
    with connect(cfg) as conn, conn.transaction():
        conn.execute("DELETE FROM rag.indexes WHERE index_version = %s", (version,))
        conn.execute(
            "INSERT INTO rag.indexes (index_version, corpus_version, chunk_size, chunk_overlap,"
            " embedding_model, n_documents, n_chunks) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (version, settings.corpus_version(cfg), r["chunk_size"], r["chunk_overlap"],
             r["embedding_model"], len(docs), len(chunks)),
        )  # fmt: skip
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO rag.chunks (index_version, chunk_id, doc_id, title, url, text,"
                " embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(version, c["chunk_id"], c["doc_id"], c["title"], c["url"], c["text"], v)
                 for c, v in zip(chunks, vectors, strict=True)],
            )  # fmt: skip
    return {"index_version": version, "documents": len(docs), "chunks": len(chunks)}


def describe(cfg: dict) -> dict | None:
    with connect(cfg) as conn:
        row = conn.execute(
            "SELECT index_version, n_documents, n_chunks, built_at FROM rag.indexes"
            " WHERE index_version = %s",
            (settings.index_version(cfg),),
        ).fetchone()
    if row is None:
        return None
    return dict(zip(("index_version", "documents", "chunks", "built_at"), row, strict=True))


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str
    distance: float


class Retriever:
    """The query path's first box: question -> top-k chunks of the configured index."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.version = settings.index_version(cfg)
        self.model = cfg["retrieval"]["embedding_model"]
        self.conn = connect(cfg)
        if describe(cfg) is None:
            raise RuntimeError(
                f"index {self.version} is not built for the current config and corpus;"
                " run `make rag-index` (or pass the same --set overrides to the index command)"
            )

    def search(self, question: str, k: int) -> list[Hit]:
        q = embed(self.model, [question])[0]
        rows = self.conn.execute(
            "SELECT chunk_id, doc_id, title, url, text, embedding <=> %s AS distance"
            " FROM rag.chunks WHERE index_version = %s ORDER BY distance LIMIT %s",
            (q, self.version, k),
        ).fetchall()
        return [Hit(*row) for row in rows]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    cfg = settings.load(overrides=args.set)
    r = cfg["retrieval"]
    print(
        f"corpus {r['corpus']} (md5 {settings.corpus_version(cfg)}), chunk_size {r['chunk_size']},"
        f" overlap {r['chunk_overlap']}, embedding {r['embedding_model']}"
    )
    out = build(cfg)
    print(
        f"index {out['index_version']}: {out['documents']} documents, {out['chunks']} chunks"
        " -> rag.chunks"
    )


if __name__ == "__main__":
    main()
