"""The prediction log: every request the routing service serves, one row, in the warehouse.

This table is the current window of the drift job. It stores the scrubbed narrative (never the
raw request body, design 6.5), the model inputs, the text descriptors, the prediction and the
model version and image commit that produced it, so a wrong prediction can be walked back to a
registry version and a line of code.

serving.labels is the case-closure feed (design 6.8): {complaint_id, final_queue, closed_at},
arriving 15-60 days after the prediction. The drift job joins it to compute quality.
"""

import json
import os

import psycopg

from complaint_routing.features import mix_columns

DDL = """
CREATE SCHEMA IF NOT EXISTS serving;
CREATE TABLE IF NOT EXISTS serving.predictions (
    request_id uuid PRIMARY KEY,
    ts timestamptz NOT NULL DEFAULT now(),
    model_name text NOT NULL,
    model_version text NOT NULL,
    image_commit text NOT NULL,
    complaint_id bigint,
    date_received timestamp,
    narrative text NOT NULL,
    company text NOT NULL,
    state text NOT NULL,
    submitted_via text NOT NULL,
    tags text NOT NULL,
    narrative_len int NOT NULL,
    n_words int NOT NULL,
    mask_share double precision NOT NULL,
    oov_share double precision NOT NULL,
    {mix},
    predicted_queue text NOT NULL,
    max_proba double precision NOT NULL,
    probabilities jsonb NOT NULL,
    latency_ms double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_ts ON serving.predictions (ts);
CREATE TABLE IF NOT EXISTS serving.labels (
    complaint_id bigint PRIMARY KEY,
    final_queue text NOT NULL,
    closed_at timestamptz NOT NULL DEFAULT now()
);
"""


def dsn() -> str:
    return os.environ["WAREHOUSE_DSN"]


def ensure_schema(conn: psycopg.Connection, queues: list[str]) -> None:
    mix = ",\n    ".join(
        f"{c} {'int' if c == 'mix_n' else 'double precision'} NOT NULL" for c in mix_columns(queues)
    )
    conn.execute(DDL.format(mix=mix))
    conn.commit()


def insert(conn: psycopg.Connection, row: dict) -> None:
    row = dict(row, probabilities=json.dumps(row["probabilities"]))
    cols = list(row)
    conn.execute(
        f"INSERT INTO serving.predictions ({', '.join(cols)})"
        f" VALUES ({', '.join(['%s'] * len(cols))})",
        [row[c] for c in cols],
    )
    conn.commit()
