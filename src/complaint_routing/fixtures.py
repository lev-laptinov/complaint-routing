"""A deliberately broken batch for the quality-gate demo (make bad-batch).

Starts from the real 2026-05 raw file and breaks it in three ways the gate must catch:
20 % of narratives set to null, 50 rows with a product missing from the mapping table,
and 100 rows dated in the previous month.
"""

import pandas as pd

from complaint_routing import config

OUT = config.DATA_DIR / "fixtures" / "bad_batch_2026-05.parquet"


def main() -> None:
    df = pd.read_parquet(config.raw_path("2026-05"))
    df.loc[df.index[:300], "Consumer complaint narrative"] = None
    df.loc[df.index[300:350], "Product"] = "Crypto lending"
    df.loc[df.index[350:450], "Date received"] = "2026-04-30"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(config.ROOT)}: {len(df)} rows")


if __name__ == "__main__":
    main()
