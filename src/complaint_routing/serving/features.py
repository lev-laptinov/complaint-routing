"""What the routing service computes per request, shared with the monitoring reference.

Two kinds of columns:
- model inputs: the request fields plus `company_queue_mix`, looked up point-in-time;
- monitoring columns derived from the text (length, words, mask share, out-of-vocabulary share),
  which the model does not use but which describe the text it sees. The drift job compares them
  between the reference window and the prediction log, so both are computed by `describe_text`.
"""

import bisect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from complaint_routing.features import mix_columns
from complaint_routing.transform import mask_share

TEXT_COLUMNS = ["narrative_len", "n_words", "mask_share", "oov_share"]
REQUEST_CATEGORICAL = ["company", "state", "submitted_via", "tags"]


def text_vocabulary(model) -> set[str]:
    """Unigrams the registered pipeline's TF-IDF vectorizer knows."""
    vec = model.named_steps["features"].named_transformers_["text"]
    return {t for t in vec.vocabulary_ if " " not in t}


def text_analyzer(model):
    vec = model.named_steps["features"].named_transformers_["text"]
    pre, tok = vec.build_preprocessor(), vec.build_tokenizer()
    return lambda text: tok(pre(text))


def describe_text(narratives: pd.Series, vocab: set[str], analyze) -> pd.DataFrame:
    def oov(text: str) -> float:
        tokens = analyze(text)
        return sum(t not in vocab for t in tokens) / max(len(tokens), 1)

    return pd.DataFrame(
        {
            "narrative_len": narratives.str.len().astype("int64"),
            "n_words": narratives.str.split().str.len().fillna(0).astype("int64"),
            "mask_share": narratives.map(mask_share).astype("float64"),
            "oov_share": narratives.map(oov).astype("float64"),
        },
        index=narratives.index,
    )


@dataclass
class MixLookup:
    """Point-in-time `company_queue_mix` for the serving path.

    Holds every row of features.complaint_features (one per complaint, event_timestamp = the
    day it was received; its mix covers [d - 150, d - 60) days) and answers "the latest row of
    this company at or before t". That is what the Feast online store holds at time t after a
    materialization up to t (Lab 2, 3). Live requests use t = now; a replayed complaint passes
    its original date, so the replayed month sees the features it would have seen that day.
    A company with no row gets zeros and mix_n = 0, as in training.
    """

    queues: list[str]
    _ts: dict[str, list[np.datetime64]] = field(default_factory=dict)
    _rows: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return mix_columns(self.queues)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, queues: list[str]) -> "MixLookup":
        out = cls(queues)
        cols = out.columns
        df = df.sort_values(["company", "event_timestamp"])
        for company, g in df.groupby("company", sort=False):
            out._ts[company] = list(g["event_timestamp"].to_numpy(dtype="datetime64[ns]"))
            out._rows[company] = g[cols].to_numpy(dtype="float64")
        return out

    def __len__(self) -> int:
        return len(self._ts)

    def get(self, company: str, at: pd.Timestamp) -> dict[str, float]:
        ts = self._ts.get(company)
        i = -1 if ts is None else bisect.bisect_right(ts, np.datetime64(at, "ns")) - 1
        values = np.zeros(len(self.columns)) if i < 0 else self._rows[company][i]
        out = dict(zip(self.columns, values.tolist(), strict=True))
        out["mix_n"] = int(out["mix_n"])
        return out


def load_mix_lookup(conn, queues: list[str], feature_version: str) -> MixLookup:
    cols = ", ".join(mix_columns(queues))
    cur = conn.execute(
        f"SELECT company, event_timestamp, {cols} FROM features.complaint_features"
        " WHERE feature_version = %s",
        (feature_version,),
    )
    df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"]).dt.tz_localize(None)
    return MixLookup.from_frame(df, queues)
