"""The two lineage identifiers: the DVC hash of the raw data and the git commit of the code."""

import hashlib
import subprocess
from pathlib import Path

import yaml

from complaint_routing.config import RAW_DVC_FILE, ROOT


def data_version(dvc_file: Path = RAW_DVC_FILE) -> str:
    """md5 of the data/raw directory as recorded by `dvc add` (ends with .dir)."""
    return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def git_commit(root: Path = ROOT) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_dirty(root: Path = ROOT) -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    return bool(out.strip())
