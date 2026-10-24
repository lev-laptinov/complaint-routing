"""company_queue_mix: the share of each queue among a company's past complaints, point-in-time.

For a complaint received on day d the window is [d - LAG - WINDOW, d - LAG): only complaints
whose label (final queue) is already known at d. Labels arrive up to 60 days after receipt
(design 5.2), so LAG = 60 days; using [d - 90, d) would read labels that do not exist yet.
"""

import pandas as pd

FEATURE_VERSION = "v1"
WINDOW_DAYS = 90
LAG_DAYS = 60


def mix_columns(queues: list[str]) -> list[str]:
    return [f"mix_{q}" for q in queues] + ["mix_n"]


def company_queue_mix(
    history: pd.DataFrame, targets: pd.DataFrame, queues: list[str]
) -> pd.DataFrame:
    """history: complaint_id, company, date_received, queue (labelled past).
    targets: complaint_id, company, date_received. Returns targets + mix columns."""
    h = history[["company", "date_received", "queue"]].copy()
    onehot = pd.get_dummies(h["queue"]).reindex(columns=queues, fill_value=0).astype("int64")
    h = pd.concat([h[["company", "date_received"]], onehot], axis=1)
    # One row per company and day, then running totals: cum[d] = complaints received before d+1.
    daily = h.groupby(["company", "date_received"], as_index=False)[queues].sum()
    daily = daily.sort_values(["company", "date_received"])
    daily[queues] = daily.groupby("company")[queues].cumsum()
    daily = daily.sort_values("date_received")

    t = targets[["complaint_id", "company", "date_received"]].copy()
    t["_hi"] = t["date_received"] - pd.Timedelta(days=LAG_DAYS)
    t["_lo"] = t["_hi"] - pd.Timedelta(days=WINDOW_DAYS)

    def cum_before(col: str) -> pd.DataFrame:
        left = t[["complaint_id", "company", col]].sort_values(col)
        right = daily.rename(columns={"date_received": col})
        m = pd.merge_asof(left, right, on=col, by="company", allow_exact_matches=False)
        return m.set_index("complaint_id")[queues].fillna(0)

    counts = (cum_before("_hi") - cum_before("_lo")).reindex(t["complaint_id"])
    total = counts.sum(axis=1)
    shares = counts.div(total.where(total > 0), axis=0).fillna(0.0)
    shares.columns = [f"mix_{q}" for q in queues]
    shares["mix_n"] = total.astype("int64")
    out = targets.merge(shares, left_on="complaint_id", right_index=True, how="left")
    out["feature_version"] = FEATURE_VERSION
    return out
