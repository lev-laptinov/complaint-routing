"""Raw rows (as delivered) -> clean typed rows. Shared by the Airflow DAG and by training."""

import re

import pandas as pd

RENAME = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
    "Submitted via": "submitted_via",
    "Tags": "tags",
}
RAW_COLUMNS = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company", "State",
    "ZIP code", "Tags", "Submitted via", "Date sent to company",
    "Company response to consumer", "Timely response?", "Complaint ID",
]  # fmt: skip
CLEAN_COLUMNS = [
    "complaint_id", "date_received", "batch_month", "product", "queue", "narrative",
    "company", "state", "submitted_via", "tags", "narrative_len", "mask_share",
]  # fmt: skip
MIN_NARRATIVE_CHARS = 50

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
_LONG_NUMBER = re.compile(r"\b\d{6,}\b")  # account, card and SSN-like numbers
_MASK = re.compile(r"X{2,}")


def scrub_pii(text: str) -> str:
    """Mask what CFPB's scrubber masks, so serving text (raw) and archive text look alike."""
    text = _EMAIL.sub("XXXX", text)
    text = _PHONE.sub("XXXX", text)
    return _LONG_NUMBER.sub("XXXX", text)


def mask_share(text: str) -> float:
    words = text.split()
    return sum(bool(_MASK.search(w)) for w in words) / max(len(words), 1)


def clean(raw: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Returns the clean rows and how many rows each cleaning rule removed."""
    df = raw[list(RENAME)].rename(columns=RENAME).copy()
    stats = {"input_rows": len(df)}

    df["complaint_id"] = pd.to_numeric(df["complaint_id"], errors="coerce").astype("Int64")
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["narrative"] = df["narrative"].fillna("").str.strip().map(scrub_pii)
    df["queue"] = df["product"].map(mapping)

    rules = {
        "bad_id_or_date": df["complaint_id"].isna() | df["date_received"].isna(),
        "unknown_product": df["queue"].isna(),
        "short_narrative": df["narrative"].str.len() < MIN_NARRATIVE_CHARS,
    }
    for name, mask in rules.items():
        stats[f"dropped_{name}"] = int(mask.sum())
        df = df[~mask.reindex(df.index, fill_value=False)]

    df = df.sort_values("complaint_id")
    before = len(df)
    df = df.drop_duplicates("complaint_id")
    stats["dropped_duplicate_id"] = before - len(df)
    before = len(df)
    df = df.drop_duplicates("narrative")  # pasted boilerplate, same text filed many times
    stats["dropped_duplicate_narrative"] = before - len(df)

    df["batch_month"] = df["date_received"].dt.strftime("%Y-%m")
    df["narrative_len"] = df["narrative"].str.len()
    df["mask_share"] = df["narrative"].map(mask_share)
    for col in ("company", "state", "submitted_via", "tags"):
        df[col] = df[col].fillna("none")
    stats["output_rows"] = len(df)
    return df[CLEAN_COLUMNS].reset_index(drop=True), stats
