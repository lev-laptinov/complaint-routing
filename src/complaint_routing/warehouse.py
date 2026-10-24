"""PostgreSQL warehouse: raw, dwh and features schemas, one partition per batch month."""

import os

import pandas as pd
import sqlalchemy as sa

TABLES = {
    "raw": "raw.complaints",
    "dwh": "dwh.complaints",
    "features": "features.complaint_features",
}


def engine() -> sa.engine.Engine:
    return sa.create_engine(os.environ["WAREHOUSE_URL"])


def replace_month(eng: sa.engine.Engine, table: str, month: str, df: pd.DataFrame) -> int:
    """Delete-then-insert of one batch month in a single transaction.

    This is what makes a rerun of the same interval harmless: the second run deletes what the
    first one wrote for that month and writes the same rows again, other months untouched."""
    schema, name = table.split(".")
    with eng.begin() as conn:
        conn.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        if sa.inspect(conn).has_table(name, schema=schema):
            conn.execute(sa.text(f"DELETE FROM {table} WHERE batch_month = :m"), {"m": month})
        df.to_sql(name, conn, schema=schema, if_exists="append", index=False, chunksize=2000)
        conn.execute(sa.text(f"CREATE INDEX IF NOT EXISTS {name}_month ON {table} (batch_month)"))
        return conn.execute(
            sa.text(f"SELECT count(*) FROM {table} WHERE batch_month = :m"), {"m": month}
        ).scalar_one()


def read_sql(eng: sa.engine.Engine, query: str, **params) -> pd.DataFrame:
    with eng.connect() as conn:
        return pd.read_sql(sa.text(query), conn, params=params)
