"""Quality gate. Each check returns a list of human-readable failures; empty means pass."""

import pandas as pd

from complaint_routing.transform import RAW_COLUMNS

MIN_ROWS = 500
MAX_NULL_NARRATIVE_SHARE = 0.05
MAX_DUPLICATE_ID_SHARE = 0.0


def validate_raw(raw: pd.DataFrame, month: str, mapping: dict[str, str]) -> list[str]:
    """Contract of a raw batch, checked before anything is transformed."""
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        return [f"schema: missing columns {missing}"]
    failures = []
    if len(raw) < MIN_ROWS:
        failures.append(f"row count {len(raw)} < {MIN_ROWS}")
    null_share = raw["Consumer complaint narrative"].isna().mean()
    if null_share > MAX_NULL_NARRATIVE_SHARE:
        failures.append(f"null narratives {null_share:.1%} > {MAX_NULL_NARRATIVE_SHARE:.0%}")
    unknown = sorted(set(raw["Product"].dropna()) - set(mapping))
    if unknown:
        failures.append(f"unknown products (add them to config/queue_mapping.csv): {unknown}")
    if raw["Product"].isna().any():
        failures.append(f"null product in {int(raw['Product'].isna().sum())} rows")
    dates = pd.to_datetime(raw["Date received"], errors="coerce")
    outside = int((dates.dt.strftime("%Y-%m") != month).sum())
    if outside:
        failures.append(f"{outside} rows with date_received outside {month} (or unparseable)")
    dup = raw["Complaint ID"].duplicated().mean()
    if dup > MAX_DUPLICATE_ID_SHARE:
        failures.append(f"duplicate complaint ids {dup:.1%}")
    return failures


def check_loaded(month: str, expected_rows: int, counts: dict[str, int], feature_version: str,
                 found_versions: list[str], null_queue: int) -> list[str]:  # fmt: skip
    """After load: the warehouse holds exactly this batch, once."""
    failures = []
    if counts["dwh"] != expected_rows:
        failures.append(
            f"dwh has {counts['dwh']} rows for {month}, transform produced {expected_rows}"
        )
    if counts["features"] != counts["dwh"]:
        failures.append(f"features {counts['features']} rows != dwh {counts['dwh']} rows")
    if counts["outside_month"]:
        failures.append(f"{counts['outside_month']} dwh rows of {month} dated outside the month")
    if null_queue:
        failures.append(f"{null_queue} rows without a queue")
    if found_versions != [feature_version]:
        failures.append(f"feature_version {found_versions} != expected [{feature_version}]")
    return failures
