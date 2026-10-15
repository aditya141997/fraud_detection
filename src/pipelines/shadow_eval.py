"""Compare champion and challenger on the same live events once their labels are known.

The scoring API (SHADOW_ALIAS=challenger) writes both models' scores and decisions into `scored-events`;
this reads them back, joins the fraud labels from the replay split (standing in for chargebacks that arrive
later) and computes, per model: PR-AUC, flag rate and expected cost (src.training.cost_matrix, with each
model's own threshold). The cost difference gets a paired bootstrap interval: the same resampled events are
scored under both models, so the interval reflects the gap, not each model's own noise.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data import PROCESSED
from src.training.cost_matrix import REVIEW_COST


def read_events(bootstrap: str, shadow_version: str, topic: str = "scored-events", idle_s: float = 5.0,
                group: str | None = None) -> pd.DataFrame:
    """All events on the topic that carry a shadow score from `shadow_version` (last one per transaction)."""
    from confluent_kafka import Consumer
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": group or f"shadow-eval-{int(time.time())}",
                  "auto.offset.reset": "earliest", "enable.auto.commit": False})
    c.subscribe([topic])
    rows, last = [], time.time()
    try:
        while time.time() - last < idle_s:
            m = c.poll(1.0)
            if m is None or m.error():
                continue
            last = time.time()
            e = json.loads(m.value())
            if str(e.get("shadow_version")) == str(shadow_version):
                rows.append({"TransactionID": e["TransactionID"], "champ_score": e["score"],
                             "champ_flag": e["flagged"], "champ_version": e["model_version"],
                             "shadow_score": e["shadow_score"], "shadow_flag": e["shadow_flagged"],
                             "amount": e["features"]["TransactionAmt"]})
    finally:
        c.close()
    return pd.DataFrame(rows).drop_duplicates("TransactionID", keep="last")


def join_labels(events: pd.DataFrame) -> pd.DataFrame:
    lab = pd.read_parquet(PROCESSED / "replay.parquet", columns=["TransactionID", "isFraud"])
    return events.merge(lab, on="TransactionID", how="inner").rename(columns={"isFraud": "y"})


def _cost(y: np.ndarray, flag: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Per-event cost: a missed fraud costs its amount, a false alarm the review fee."""
    return np.where((y == 1) & ~flag, amount, 0.0) + np.where((y == 0) & flag, REVIEW_COST, 0.0)


def compare(df: pd.DataFrame, n_boot: int = 2000, confidence: float = 0.95, seed: int = 0,
            retuned_threshold: float | None = None) -> dict:
    """df columns: y, amount, champ_score, champ_flag, shadow_score, shadow_flag (one row per event).
    `retuned_threshold` adds an informational third arm: the champion with only its threshold re-tuned on
    recent labels, i.e. the cheaper alternative to retraining. The promotion rule does not use it."""
    y, amt = df["y"].to_numpy(), df["amount"].to_numpy(dtype=float)
    cf, sf = df["champ_flag"].to_numpy(bool), df["shadow_flag"].to_numpy(bool)
    cc, cs = _cost(y, cf, amt), _cost(y, sf, amt)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=(n_boot, len(df)))
    saving = 1 - cs[idx].sum(axis=1) / cc[idx].sum(axis=1)
    lo, hi = np.quantile(saving, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])

    def arm(score, flag, cost):
        return {"pr_auc": float(average_precision_score(y, score)), "flag_rate": float(flag.mean()),
                "expected_cost": float(cost.sum())}

    out = {"n_events": len(df), "n_frauds": int(y.sum()),
           "champion": arm(df["champ_score"], cf, cc), "challenger": arm(df["shadow_score"], sf, cs),
           "cost_saving": float(1 - cs.sum() / cc.sum()), "cost_saving_ci": [float(lo), float(hi)],
           "confidence": confidence, "no_model_cost": float(amt[y == 1].sum())}
    if retuned_threshold is not None:
        rf = (df["champ_score"] >= retuned_threshold).to_numpy()
        cr = _cost(y, rf, amt)
        out["champion_retuned"] = {**arm(df["champ_score"], rf, cr), "threshold": retuned_threshold,
                                   "saving_vs_champion": float(1 - cr.sum() / cc.sum())}
    return out
