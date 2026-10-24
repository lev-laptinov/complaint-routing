"""Build the raw layer: one Parquet sample per month from the public CFPB narratives archive.

    uv run python -m complaint_routing.ingest            # download (if missing) + sample
    uv run python -m complaint_routing.ingest --verify   # rebuild and compare with data/raw.dvc

Sampling rule (documented in the README): among complaints of the month that have a published
narrative, deduplicated by Complaint ID and sorted by it, draw SAMPLE_PER_MONTH rows with
pandas' sample(random_state=YYYYMM). Columns are kept exactly as delivered (all strings).
"""

import argparse
import subprocess
import urllib.request
import zipfile

import pandas as pd

from complaint_routing.config import (
    FIRST_MONTH,
    LAST_MONTH,
    RAW_DIR,
    SAMPLE_PER_MONTH,
    SOURCE_DIR,
    raw_path,
)

BASE_URL = "https://files.consumerfinance.gov/f/documents/"
# Exports that cover FIRST_MONTH..LAST_MONTH (FOIA reading room, checked 2026-10-24).
ARCHIVE_FILES = [
    "CCDB_Export_3_May_2021_through_October_2022.zip",
    "CCDB_Export_4_November_2022_through_August_2023.zip",
    "CCDB_Export_5_September_2023_through_March_2024.zip",
    "CCDB_Export_6_April_2024_through_July_2024.zip",
    "CCDB_Export_7_August_2024_through_October_2024.zip",
    "CCDB_Export_8_November_2024_through_December_2024.zip",
    "CCDB_Export_9_January_2025_through_February_2025.zip",
    "CCDB_Export_10_March_2025_through_April_2025.zip",
    "CCDB_Export_11_May_2025_through_June_2025.zip",
    "CCDB_Export_12_July_2025_through_August_2025.zip",
    "CCDB_Export_13_September_2025_through_October_2025.zip",
    "CCDB_Export_14_November_2025_through_December_2025.zip",
    "CCDB_Export_15_January_2026_through_February_2026.zip",
    "CCDB_Export_16_March_2026.zip",
    "CCDB_Export_17_April_2026.zip",
    "CCDB_Export_18_May_2026.zip",
    "CCDB_Export_19_June_2026.zip",
    "CCDB_Export_20_July_2026.zip",
]
NARRATIVE = "Consumer complaint narrative"


def download_archive() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVE_FILES:
        dest = SOURCE_DIR / name
        if dest.exists():
            continue
        print(f"downloading {name}", flush=True)
        # The CDN rejects browser-like user agents without browser headers; curl's default passes.
        req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "curl/8.7.1"})
        with urllib.request.urlopen(req, timeout=300) as r, open(f"{dest}.part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        (SOURCE_DIR / f"{name}.part").rename(dest)


def read_narrative_rows() -> pd.DataFrame:
    parts = []
    for name in ARCHIVE_FILES:
        with zipfile.ZipFile(SOURCE_DIR / name) as z:
            csv_name = z.namelist()[0]
            for chunk in pd.read_csv(z.open(csv_name), dtype=str, chunksize=500_000):
                month = chunk["Date received"].str[:7]
                keep = chunk[NARRATIVE].notna() & month.between(FIRST_MONTH, LAST_MONTH)
                parts.append(chunk[keep])
        print(f"read {name}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("Complaint ID")
    return df


def sample_month(df: pd.DataFrame, month: str) -> pd.DataFrame:
    rows = df[df["Date received"].str[:7] == month].copy()
    rows["_id"] = rows["Complaint ID"].astype("int64")
    rows = rows.sort_values("_id")
    n = min(SAMPLE_PER_MONTH, len(rows))
    sample = rows.sample(n=n, random_state=int(month.replace("-", "")))
    return sample.sort_values("_id").drop(columns="_id").reset_index(drop=True)


def build_raw() -> None:
    download_archive()
    df = read_narrative_rows()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for month in sorted(df["Date received"].str[:7].unique()):
        sample = sample_month(df, month)
        sample.to_parquet(raw_path(month), index=False, compression="zstd")
        print(f"{month}: {len(sample)} rows", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--verify", action="store_true", help="after rebuilding, check against data/raw.dvc"
    )
    args = p.parse_args()
    build_raw()
    if args.verify:
        # `dvc status` compares the workspace with the hashes recorded in data/raw.dvc.
        subprocess.run(["dvc", "status", "data/raw.dvc"], check=True)


if __name__ == "__main__":
    main()
