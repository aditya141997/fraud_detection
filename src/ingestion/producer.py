"""Replay the held-out stream into Redpanda at a controllable rate.

    python -m src.ingestion.producer --rate 200 --limit 5000
"""
import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer

from src.data import PROCESSED

TOPIC = "transactions"


def to_event(rec: dict) -> bytes:
    return json.dumps({k: v for k, v in rec.items() if v == v and v is not None}).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=200, help="events per second (0 = as fast as possible)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", default="replay")
    ap.add_argument("--bootstrap", default="localhost:19092")
    a = ap.parse_args()

    df = pd.read_parquet(PROCESSED / f"{a.source}.parquet")
    if a.limit:
        df = df.head(a.limit)
    p = Producer({"bootstrap.servers": a.bootstrap, "linger.ms": 5})
    start = time.perf_counter()
    for i, rec in enumerate(df.to_dict("records"), 1):
        p.produce(TOPIC, key=str(rec["TransactionID"]), value=to_event(rec))
        p.poll(0)
        if a.rate and i % 10 == 0:
            lag = i / a.rate - (time.perf_counter() - start)
            if lag > 0:
                time.sleep(lag)
    p.flush()
    print(f"sent {len(df):,} events in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
