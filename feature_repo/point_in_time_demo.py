"""Point-in-time retrieval: an entity timestamp between two feature rows returns the earlier row.

make feast-demo
"""

import os

import pandas as pd
import psycopg
from feast import FeatureStore

COMPANY = "CAPITAL ONE FINANCIAL CORPORATION"
FEATURES = ["company_queue_mix:mix_card", "company_queue_mix:mix_checking_savings",
            "company_queue_mix:mix_n"]  # fmt: skip


def two_consecutive_rows() -> tuple[pd.Series, pd.Series]:
    """The first pair of consecutive feature rows of COMPANY in 2024-03 whose values differ."""
    user, password = os.environ["POSTGRES_USER"], os.environ["POSTGRES_PASSWORD"]
    with psycopg.connect(
        host="localhost", port=5433, dbname="warehouse", user=user, password=password
    ) as conn:
        cur = conn.execute(
            "SELECT DISTINCT event_timestamp, mix_card, mix_checking_savings, mix_n "
            "FROM features.complaint_features WHERE company = %s AND batch_month = '2024-03' "
            "ORDER BY event_timestamp",
            (COMPANY,),
        )
        rows = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    i = rows.index[rows["mix_n"].diff().fillna(0) != 0][0]
    return rows.loc[i - 1], rows.loc[i]


def main() -> None:
    earlier, later = two_consecutive_rows()
    between = (
        earlier["event_timestamp"] + (later["event_timestamp"] - earlier["event_timestamp"]) / 2
    )
    print(f"two feature rows of {COMPANY}:")
    print(pd.DataFrame([earlier, later]).to_string(index=False))

    store = FeatureStore(repo_path=".")
    entity_df = pd.DataFrame(
        {"company": [COMPANY], "event_timestamp": [pd.Timestamp(between, tz="UTC")]}
    )
    got = store.get_historical_features(entity_df=entity_df, features=FEATURES).to_df()
    print(f"\nget_historical_features at {between} (between the two rows):")
    print(got.to_string(index=False))

    assert got["mix_n"].iloc[0] == earlier["mix_n"] != later["mix_n"]
    print("\nOK: Feast returned the earlier row; the later one is after the entity timestamp.")


if __name__ == "__main__":
    main()
