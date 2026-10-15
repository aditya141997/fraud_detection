"""Infrastructure adapters for the scoring service: Feast/Redis feature lookup and Kafka event publishing."""
import json
import logging
import time
from pathlib import Path

from confluent_kafka import Producer
from feast import FeatureStore

from src.serving.scorer import JOIN_KEY, VIEW_COLUMNS

log = logging.getLogger("fraud.serving")
REPO = Path(__file__).resolve().parents[2] / "feature_repo"
SCORED_TOPIC = "scored-events"


class FeastFetcher:
    """One Feast call per request, asking only for the views whose entity key the transaction has."""

    def __init__(self, repo: Path = REPO):
        self.fs = FeatureStore(str(repo))

    def __call__(self, keys: dict) -> tuple[dict, list]:
        if not keys:
            return {}, []
        features = [f"{p}_velocity:{c}" for p in keys for c in VIEW_COLUMNS[p]]
        row = {JOIN_KEY[p]: k for p, k in keys.items()}
        out = self.fs.get_online_features(features=features, entity_rows=[row]).to_dict()
        feats = {c: out[c][0] for p in keys for c in VIEW_COLUMNS[p]}
        # an entity seen by the API but never written by the consumer comes back as None
        missing = [p for p in keys if any(feats[c] is None for c in VIEW_COLUMNS[p])]
        return feats, missing


class EventPublisher:
    """Fire-and-forget publisher to the scored-events topic. Never blocks or fails a /score call."""

    def __init__(self, bootstrap: str, monitored: list[str]):
        self.monitored = monitored
        self.dropped = 0
        self.producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 20,
                                  "message.timeout.ms": 5000, "queue.buffering.max.messages": 100_000})

    def publish(self, txn: dict, result: dict) -> None:
        feats = result["features"]
        event = {
            "TransactionID": txn.get("TransactionID"), "TransactionDT": txn.get("TransactionDT"),
            "scored_at": time.time(), "score": result["score"], "flagged": result["flagged"],
            "model_version": result["model_version"],
            "features": {c: txn.get(c, feats.get(c)) for c in self.monitored},
        }
        if "shadow" in result:  # challenger's verdict on the same event, for offline comparison with labels
            event.update(shadow_score=result["shadow"]["score"], shadow_flagged=result["shadow"]["flagged"],
                         shadow_version=result["shadow"]["model_version"],
                         shadow_threshold=result["shadow"]["threshold"])
        event["threshold"] = result["threshold"]
        try:
            self.producer.produce(SCORED_TOPIC, json.dumps(event, default=str).encode(),
                                  on_delivery=self._delivered)
            self.producer.poll(0)
        except BufferError:
            self.dropped += 1

    def _delivered(self, err, _msg):
        if err is not None:
            self.dropped += 1
            log.warning("scored-event delivery failed: %s", err)

    def close(self, timeout: float = 5.0) -> None:
        self.producer.flush(timeout)


class NullPublisher:
    dropped = 0

    def publish(self, txn, result):
        pass

    def close(self, timeout=0):
        pass
