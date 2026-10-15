from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64, Int64
from feast.value_type import ValueType

card = Entity(name="card", join_keys=["card_id"], value_type=ValueType.STRING)
device = Entity(name="device", join_keys=["device_info"], value_type=ValueType.STRING)
email = Entity(name="email", join_keys=["email_domain"], value_type=ValueType.STRING)


def _view(prefix: str, entity: Entity, key: str) -> FeatureView:
    source = FileSource(
        path=f"../data/processed/features_{prefix}.parquet",
        timestamp_field="event_timestamp",
    )
    schema = []
    for w in ("1h", "24h"):
        schema += [
            Field(name=f"{prefix}_txn_count_{w}", dtype=Int64),
            Field(name=f"{prefix}_amt_sum_{w}", dtype=Float64),
        ]
    return FeatureView(
        name=f"{prefix}_velocity",
        entities=[entity],
        ttl=timedelta(days=2),
        schema=schema,
        source=source,
    )


card_velocity = _view("card", card, "card_id")
device_velocity = _view("device", device, "device_info")
email_velocity = _view("email", email, "email_domain")

fraud_features_v1 = FeatureService(
    name="fraud_features_v1",
    features=[card_velocity, device_velocity, email_velocity],
)
