"""Replay the held-out stream through the live API, optionally with injected drift from a given event.

Same lock-step as src.serving.replay_score (update rolling state, write features to Redis, call /score) but
without the parity check, and with `apply_drift` applied to every event from --drift-start on.

    python -m src.pipelines.stream --limit 40000 --drift-start 8000 [--rate 0]
Flush Redis first (see README): Feast ignores writes that are not newer than what it holds.
"""
import argparse
import time

import httpx
import pandas as pd
from feast import FeatureStore

from src.data import PROCESSED
from src.pipelines import scenario
from src.pipelines.drift_injection import DriftSpec, apply_drift, top_numeric_features
from src.serving.adapters import REPO
from src.serving.registry import load_alias
from src.serving.replay_score import payload, warm_state, write_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--start", type=int, default=0, help="first event to score; earlier ones only warm the state")
    ap.add_argument("--limit", type=int, default=scenario.SHADOW_END, help="stream events [start, limit)")
    ap.add_argument("--drift-start", type=int, default=scenario.DRIFT_START)
    ap.add_argument("--amount-scale", type=float, default=DriftSpec.amount_scale)
    ap.add_argument("--feature-scale", type=float, default=DriftSpec.feature_scale)
    ap.add_argument("--rate", type=float, default=0, help="events/sec, 0 = as fast as possible")
    a = ap.parse_args()

    booster, _, _ = load_alias("champion")
    spec = DriftSpec(a.amount_scale, a.feature_scale, top_numeric_features(booster))
    print(f"drift from event {a.drift_start}: {spec}", flush=True)

    full = pd.read_parquet(PROCESSED / "replay.parquet").reset_index(drop=True).iloc[:a.limit]
    full = pd.concat([full.iloc[:a.drift_start], apply_drift(full.iloc[a.drift_start:], spec)])
    fs, state = FeatureStore(str(REPO)), warm_state()
    for rec in full.iloc[:a.start].to_dict("records"):  # an earlier phase already scored these: advance state only
        state.update(rec)
    full = full.iloc[a.start:]
    t0, failed, retried = time.perf_counter(), 0, 0
    with httpx.Client(base_url=a.url, timeout=30) as http:
        for i, rec in enumerate(full.to_dict("records")):
            write_features(fs, rec, state)
            for attempt in range(3):  # a slow request is retried (scoring is idempotent), never silently skipped
                try:
                    ok = http.post("/score", json=payload(rec)).status_code == 200
                    break
                except httpx.TransportError as exc:
                    retried += 1
                    print(f"event {a.start + i}: {type(exc).__name__}, retry {attempt + 1}", flush=True)
            else:
                ok = False
            failed += not ok
            if a.rate:
                time.sleep(max(0, (i + 1) / a.rate - (time.perf_counter() - t0)))
            if (i + 1) % 2000 == 0:
                print(f"streamed {i + 1:,} ({(i + 1) / (time.perf_counter() - t0):.0f}/s), failed {failed}",
                      flush=True)
    print(f"done: {len(full):,} events, {failed} failed, {retried} retried")


if __name__ == "__main__":
    main()
