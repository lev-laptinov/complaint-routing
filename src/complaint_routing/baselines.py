"""Rule-based baselines: the majority queue, and keyword rules with a majority fallback."""

import re

import numpy as np
import pandas as pd

# Ordered: the first queue whose pattern matches wins. Narrow products before broad ones.
KEYWORD_RULES = [
    (
        "student_loan",
        r"student loan|navient|nelnet|mohela|loan servicer.*(federal|education)|fafsa",
    ),
    ("mortgage", r"mortgage|escrow|foreclos|loan modification|refinanc|home loan"),
    ("vehicle_loan", r"\bcar\b|vehicle|auto loan|repossess|dealership|\blease\b"),
    ("personal_loan", r"payday|title loan|personal loan|installment loan|cash advance"),
    ("debt_management", r"debt (settlement|relief|management)|credit counsel"),
    (
        "money_transfer",
        r"zelle|venmo|paypal|cash app|wire transfer|money transfer|crypto|bitcoin|coinbase",
    ),
    (
        "debt_collection",
        r"collector|collection agenc|debt collect|\bowe\b|validation of (the )?debt|fdcpa",
    ),
    ("card", r"credit card|\bcard\b|annual fee|cash back|prepaid"),
    ("checking_savings", r"checking|savings|overdraft|bank account|deposit|\batm\b"),
    ("credit_reporting", r"credit report|equifax|experian|transunion|fcra|inquir|dispute"),
]
_COMPILED = [(q, re.compile(p, re.IGNORECASE)) for q, p in KEYWORD_RULES]


class MajorityBaseline:
    def fit(self, y: pd.Series) -> "MajorityBaseline":
        self.queue_ = y.value_counts().idxmax()
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.queue_, dtype=object)


class KeywordBaseline:
    def __init__(self, fallback: str):
        self.fallback = fallback

    def predict(self, narratives: pd.Series) -> np.ndarray:
        def route(text: str) -> str:
            for queue, pattern in _COMPILED:
                if pattern.search(text):
                    return queue
            return self.fallback

        return narratives.map(route).to_numpy(dtype=object)
