import pandas as pd

from complaint_routing import config, quality
from complaint_routing.features import company_queue_mix
from complaint_routing.transform import RAW_COLUMNS, clean

MAPPING = config.load_mapping()


def raw_rows(n: int, month: str = "2026-05") -> pd.DataFrame:
    df = pd.DataFrame({c: ["x"] * n for c in RAW_COLUMNS})
    df["Complaint ID"] = [str(1000 + i) for i in range(n)]
    df["Date received"] = f"{month}-15"
    df["Product"] = "Debt collection"
    df["Consumer complaint narrative"] = [
        f"collector keeps calling about debt number {i} " * 3 for i in range(n)
    ]
    return df


def test_validate_passes_good_batch_and_names_each_broken_rule():
    good = raw_rows(600)
    assert quality.validate_raw(good, "2026-05", MAPPING) == []
    bad = good.copy()
    bad.loc[:59, "Consumer complaint narrative"] = None
    bad.loc[60:61, "Product"] = "Crypto lending"
    bad.loc[62:63, "Date received"] = "2026-04-30"
    failures = " | ".join(quality.validate_raw(bad, "2026-05", MAPPING))
    assert (
        "null narratives" in failures
        and "Crypto lending" in failures
        and "outside 2026-05" in failures
    )


def test_clean_drops_duplicates_and_maps_queue_and_scrubs():
    raw = raw_rows(3)
    raw.loc[1, "Consumer complaint narrative"] = raw.loc[0, "Consumer complaint narrative"]
    raw.loc[2, "Consumer complaint narrative"] = (
        "please call me at 555-123-4567 about my account 123456789, the bank closed it"
    )
    df, stats = clean(raw, MAPPING)
    assert stats["dropped_duplicate_narrative"] == 1 and len(df) == 2
    assert set(df["queue"]) == {"debt_collection"}
    assert "555" not in df["narrative"].iloc[1] and "123456789" not in df["narrative"].iloc[1]


def test_company_mix_never_reads_the_future_or_unlabelled_recent_past():
    q = config.queues()
    hist = pd.DataFrame({
        "complaint_id": [1, 2, 3, 4],
        "company": ["A"] * 4,
        "date_received": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-04-20", "2026-06-01"]),
        "queue": ["mortgage", "card", "card", "card"],
    })  # fmt: skip
    target = pd.DataFrame({"complaint_id": [9], "company": ["A"],
                           "date_received": pd.to_datetime(["2026-05-01"])})  # fmt: skip
    row = company_queue_mix(hist, target, q).iloc[0]
    # window for 2026-05-01 = [2025-12-02, 2026-03-02): ids 1, 2 (3 too recent, 4 in the future)
    assert row["mix_n"] == 2 and row["mix_mortgage"] == 0.5 and row["mix_card"] == 0.5
