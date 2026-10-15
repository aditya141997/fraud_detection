"""Drive the scoring API with replayed transactions and check serving/offline parity.

For each replay event, in time order, this does what the Redpanda feature consumer does (update rolling
velocity state, write the event's features to Redis through Feast) and then calls POST /score. The API
therefore sees exactly the online features the model saw offline for that event, and its score must
equal the offline model score on the same row.

    python -m src.serving.replay_score --limit 2000 [--start 0] [--url http://localhost:8000] [--rate 0]
"""
import argparse
import math
import time

import httpx
import numpy as np
import pandas as pd
from feast import FeatureStore

from src.data import PROCESSED
from src.features import EPOCH, add_velocity
from src.ingestion.consumer import KEY_COL
from src.ingestion.velocity import MAX_WINDOW, VelocityState
from src.serving.adapters import REPO
from src.training.train import split_xy

import lightgbm as lgb

DROP_FROM_PAYLOAD = ("isFraud",)


def payload(rec: dict) -> dict:
    """JSON-safe transaction: NaN/None columns are omitted (the API treats absent columns as missing)."""
    out = {}
    for k, v in rec.items():
        if k in DROP_FROM_PAYLOAD or v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out[k] = v.item() if hasattr(v, "item") else v
    return out


def warm_state(split: str = "val") -> VelocityState:
    hist = pd.read_parquet(PROCESSED / f"{split}.parquet",
                           columns=["TransactionDT", "TransactionAmt", "card1", "card2", "card3", "card5",
                                    "DeviceInfo", "P_emaildomain"])
    hist = hist[hist.TransactionDT >= hist.TransactionDT.max() - MAX_WINDOW]
    state = VelocityState()
    for rec in hist.to_dict("records"):
        state.update(rec)
    return state


def write_features(fs: FeatureStore, rec: dict, state: VelocityState) -> None:
    ts = EPOCH + pd.Timedelta(seconds=rec["TransactionDT"])
    for prefix, (key, feats) in state.update(rec).items():
        fs.write_to_online_store(f"{prefix}_velocity",
                                 pd.DataFrame([{KEY_COL[prefix]: key, "event_timestamp": ts, **feats}]))


def offline_scores(df: pd.DataFrame, model_file: str) -> np.ndarray:
    b = lgb.Booster(model_file=model_file)
    X, _ = split_xy(add_velocity(df))
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    for c, levels in zip(cats, b.pandas_categorical):
        X[c] = pd.Categorical(X[c], categories=levels)
    return b.predict(X[b.feature_name()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--start", type=int, default=0, help="first replay event to score; earlier ones only warm the state")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--rate", type=float, default=0, help="events/sec, 0 = as fast as possible")
    ap.add_argument("--model-file", default=str(PROCESSED / "model_v1.txt"), help="offline copy of the champion")
    ap.add_argument("--no-parity", action="store_true")
    a = ap.parse_args()

    full = pd.read_parquet(PROCESSED / "replay.parquet").reset_index(drop=True)
    df = full.iloc[a.start:a.start + a.limit].reset_index(drop=True)
    fs, state = FeatureStore(str(REPO)), warm_state()
    for rec in full.iloc[:a.start].to_dict("records"):  # advance rolling windows without scoring
        state.update(rec)
    recs = df.to_dict("records")
    api = []
    with httpx.Client(base_url=a.url, timeout=10) as http:
        t0 = time.perf_counter()
        for i, rec in enumerate(recs):
            write_features(fs, rec, state)
            r = http.post("/score", json=payload(rec))
            r.raise_for_status()
            api.append(r.json())
            if a.rate:
                time.sleep(max(0, (i + 1) / a.rate - (time.perf_counter() - t0)))
            if (i + 1) % 500 == 0:
                print(f"scored {i + 1:,}", flush=True)
    scores = np.array([r["score"] for r in api])
    flagged = np.array([r["flagged"] for r in api])
    print(f"scored {len(api):,} events; flagged {flagged.mean():.1%}; "
          f"events with missing online entities: {sum(bool(r['missing_entities']) for r in api)}")
    if not a.no_parity:
        off = offline_scores(df, a.model_file)
        thr = api[0]["threshold"]
        d = np.abs(scores - off)
        print(f"parity vs offline: max abs score diff {d.max():.2e}, "
              f"decision mismatches {int(((off >= thr) != flagged).sum())} of {len(off):,}")


if __name__ == "__main__":
    main()
