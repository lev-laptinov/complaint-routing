"""The deliberate distribution shift of Lab 4, as a documented, seeded transformation.

`truncate`: in a fraction of the rows (default 30 %), the narrative is cut to its first N words
(default 25). It imitates an intake form that starts limiting the free-text box, a covariate
shift in the text the model reads: P(text) changes, the routing rule P(queue | text) does not.
"""

import numpy as np
import pandas as pd

SHIFTS = ("none", "truncate")


def truncate(df: pd.DataFrame, fraction: float = 0.3, words: int = 25, seed: int = 7):
    rng = np.random.default_rng(seed)
    hit = rng.random(len(df)) < fraction
    out = df.copy()
    out.loc[hit, "narrative"] = out.loc[hit, "narrative"].str.split().str[:words].str.join(" ")
    return out, hit


def apply(df: pd.DataFrame, shift: str, fraction: float, words: int, seed: int):
    if shift == "none":
        return df.copy(), np.zeros(len(df), dtype=bool)
    if shift == "truncate":
        return truncate(df, fraction, words, seed)
    raise ValueError(f"unknown shift {shift}; known: {SHIFTS}")
