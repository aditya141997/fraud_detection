"""Consume transactions, compute rolling velocity features, materialise them to Redis via Feast.

Each event's features (past-only) are written under that event's entity keys, so the scoring API
can read exactly what the model saw offline when it handles the same transaction.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from confluent_kafka import Consumer
from feast import FeatureStore

from src.data import PROCESSED
from src.features import EPOCH
from src.ingestion.velocity import MAX_WINDOW, VelocityState

TOPIC = "transactions"
KEY_COL = {"card": "card_id", "device": "device_info", "email": "email_domain"}


def flush(fs: FeatureStore, batch: dict) -> None:
    for prefix, rows in batch.items():
        if rows:
            fs.write_to_online_store(f"{prefix}_velocity", pd.DataFrame(rows))
            rows.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:19092")
    ap.add_argument("--group", default="velocity-consumer")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--warm-from", default="val", help="split whose last 24h primes the windows ('' = cold)")
    a = ap.parse_args()

    fs = FeatureStore(str(Path(__file__).resolve().parents[2] / "feature_repo"))
    c = Consumer({"bootstrap.servers": a.bootstrap, "group.id": a.group,
                  "auto.offset.reset": "earliest", "enable.auto.commit": True})
    c.subscribe([TOPIC])
    state, batch, n = VelocityState(), {p: [] for p in KEY_COL}, 0
    if a.warm_from:  # bootstrap windows from history so the first replayed events aren't cold
        hist = pd.read_parquet(PROCESSED / f"{a.warm_from}.parquet",
                               columns=["TransactionDT", "TransactionAmt", "card1", "card2",
                                        "card3", "card5", "DeviceInfo", "P_emaildomain"])
        hist = hist[hist.TransactionDT >= hist.TransactionDT.max() - MAX_WINDOW]
        for rec in hist.to_dict("records"):
            state.update(rec)
        print(f"warmed state with {len(hist):,} events from {a.warm_from}", flush=True)
    try:
        while not a.max_events or n < a.max_events:
            msg = c.poll(1.0)
            if msg is None:
                flush(fs, batch)  # idle: push what we have
                continue
            if msg.error():
                raise RuntimeError(msg.error())
            rec = json.loads(msg.value())
            ts = EPOCH + pd.Timedelta(seconds=rec["TransactionDT"])
            for prefix, (key, feats) in state.update(rec).items():
                batch[prefix].append({KEY_COL[prefix]: key, "event_timestamp": ts, **feats})
            n += 1
            if n % a.batch == 0:
                flush(fs, batch)
                print(f"processed {n:,}", flush=True)
    finally:
        flush(fs, batch)
        c.close()


if __name__ == "__main__":
    main()
