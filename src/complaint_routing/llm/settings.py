"""The LLM configuration file, its overrides, and the hashes that identify a run.

Three identifiers make an LLM run reproducible:
- the config hash: sha256 of the effective configuration (file + overrides), canonical JSON;
- a prompt hash per prompt file: sha256 of the file's bytes, so editing a prompt is a new run;
- the index version: corpus md5 + chunk size + overlap + embedding model, the key of the
  pgvector rows. The query path refuses to run against an index version that was not built.
"""

import hashlib
import json
import os
from pathlib import Path
from string import Template

import yaml

from complaint_routing.config import ROOT

CONFIG_PATH = ROOT / "config" / "llm.yaml"


def short_hash(data: bytes, n: int = 12) -> str:
    return hashlib.sha256(data).hexdigest()[:n]


def _coerce(value: str):
    return yaml.safe_load(value)  # "60" -> 60, "null" -> None, "0.5" -> 0.5


def load(path: Path = CONFIG_PATH, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown setting {key}")
        node[leaf] = _coerce(value)
    return cfg


def config_hash(cfg: dict) -> str:
    return short_hash(json.dumps(cfg, sort_keys=True).encode())


def prompt_path(cfg: dict, name: str) -> Path:
    return ROOT / cfg["prompts"][name]


def prompt_hash(cfg: dict, name: str) -> str:
    return short_hash(prompt_path(cfg, name).read_bytes())


def render(cfg: dict, name: str, **values: str) -> str:
    """Fill a prompt file; placeholders are $name, so JSON braces in prompts need no escaping."""
    return Template(prompt_path(cfg, name).read_text()).substitute(values)


def corpus_path(cfg: dict) -> Path:
    return ROOT / cfg["retrieval"]["corpus"]


def corpus_version(cfg: dict) -> str:
    return hashlib.md5(corpus_path(cfg).read_bytes()).hexdigest()


def index_version(cfg: dict) -> str:
    r = cfg["retrieval"]
    key = {
        "corpus": corpus_version(cfg),
        "chunk_size": r["chunk_size"],
        "chunk_overlap": r["chunk_overlap"],
        "embedding_model": r["embedding_model"],
    }
    return short_hash(json.dumps(key, sort_keys=True).encode())


def db_url(cfg: dict) -> str:
    return os.environ[cfg["retrieval"]["db_url_env"]]

