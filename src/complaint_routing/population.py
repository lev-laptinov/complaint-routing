"""Queue mix of ALL complaints in the test months (structured fields, with or without narrative).

    uv run python -m complaint_routing.population   # writes config/test_population_mix.csv

The training data only has complaints with a published narrative (credit reporting ~4 % in 2026),
an intake service sees every complaint (credit reporting ~90 %). Design 5.4 re-weights per-queue
test recall with this mix to estimate what an intake service would actually get.
"""

import zipfile

import pandas as pd

from complaint_routing import config
from complaint_routing.ingest import ARCHIVE_FILES

OUT = config.ROOT / "config" / "test_population_mix.csv"


def main() -> None:
    lo, hi = config.TEST_MONTHS
    mapping = config.load_mapping()
    counts = pd.Series(dtype="int64")
    for name in ARCHIVE_FILES:
        with zipfile.ZipFile(config.SOURCE_DIR / name) as z:
            for chunk in pd.read_csv(z.open(z.namelist()[0]), dtype=str, chunksize=500_000,
                                     usecols=["Date received", "Product"]):  # fmt: skip
                chunk = chunk[chunk["Date received"].str[:7].between(lo, hi)]
                counts = counts.add(chunk["Product"].map(mapping).value_counts(), fill_value=0)
    mix = (counts / counts.sum()).rename("share").rename_axis("queue").reset_index()
    mix.sort_values("share", ascending=False).to_csv(OUT, index=False, float_format="%.6f")
    print(
        mix.sort_values("share", ascending=False).to_string(index=False),
        f"\nrows: {int(counts.sum())}",
    )


if __name__ == "__main__":
    main()
