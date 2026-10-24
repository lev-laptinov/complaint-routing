"""Feast definitions: one entity (company) and one feature view (its point-in-time queue mix)."""

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int64

QUEUES = [
    "card", "checking_savings", "credit_reporting", "debt_collection", "debt_management",
    "money_transfer", "mortgage", "personal_loan", "student_loan", "vehicle_loan",
]  # fmt: skip

company = Entity(
    name="company", join_keys=["company"], description="Company named in the complaint"
)

# Each row: the company's queue mix as known at event_timestamp (one row per complaint).
source = PostgreSQLSource(
    name="complaint_features",
    query="SELECT company, event_timestamp, " + ", ".join(f"mix_{q}" for q in QUEUES) + ", mix_n "
    "FROM features.complaint_features WHERE feature_version = 'v1'",
    timestamp_field="event_timestamp",
)

company_queue_mix = FeatureView(
    name="company_queue_mix",
    entities=[company],
    ttl=timedelta(days=365),
    schema=[Field(name=f"mix_{q}", dtype=Float64) for q in QUEUES]
    + [Field(name="mix_n", dtype=Int64)],
    source=source,
    online=True,
    tags={"feature_version": "v1", "window_days": "90", "lag_days": "60"},
)
