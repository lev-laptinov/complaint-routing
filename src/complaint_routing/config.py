"""Paths and constants shared by the pipeline, training and reproduction."""

import csv
import hashlib
import os
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"  # downloaded CFPB zips, not versioned (rebuildable)
RAW_DIR = DATA_DIR / "raw"  # monthly samples, versioned with DVC
RAW_DVC_FILE = DATA_DIR / "raw.dvc"
MAPPING_PATH = ROOT / "config" / "queue_mapping.csv"

FIRST_MONTH = "2022-01"
LAST_MONTH = "2026-07"
SAMPLE_PER_MONTH = 1500

# Time split (design 5.4). Inclusive month ranges.
TRAIN_MONTHS = ("2022-01", "2026-02")
VAL_MONTHS = ("2026-03", "2026-04")
TEST_MONTHS = ("2026-05", "2026-07")


def raw_path(month: str) -> Path:
    return RAW_DIR / f"complaints_{month}.parquet"


def months_between(first: str, last: str) -> list[str]:
    y, m = map(int, first.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def load_mapping(path: Path = MAPPING_PATH) -> dict[str, str]:
    with open(path, newline="") as f:
        return {row["product"]: row["queue"] for row in csv.DictReader(f)}


def mapping_version(path: Path = MAPPING_PATH) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def queues(path: Path = MAPPING_PATH) -> list[str]:
    return sorted(set(load_mapping(path).values()))
