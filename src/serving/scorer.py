"""Scoring core: turn one raw transaction + online velocity features into a fraud score.

Kept free of FastAPI / Redis / Kafka so it can be unit-tested. The model is fed a plain numpy row:
categorical columns are encoded with the category lists LightGBM stored in the model file
(`pandas_categorical`), which gives byte-identical predictions to the pandas path used in training at
roughly a tenth of the per-row cost (1.5 ms vs 12.7 ms, measured on 100 single-row calls).
"""
import math
import time
from typing import Callable

import lightgbm as lgb
import numpy as np

from src.ingestion.velocity import entity_keys
from src.training.explain import top_reasons_array

# Online feature views and the columns each one provides (see feature_repo/definitions.py).
VIEW_COLUMNS = {
    p: [f"{p}_txn_count_1h", f"{p}_amt_sum_1h", f"{p}_txn_count_24h", f"{p}_amt_sum_24h"]
    for p in ("card", "device", "email")
}
JOIN_KEY = {"card": "card_id", "device": "device_info", "email": "email_domain"}

# fetch(entity -> key) -> (features, missing_entities); injected so tests need no Redis
Fetcher = Callable[[dict], tuple[dict, list]]


class Scorer:
    def __init__(self, booster: lgb.Booster, threshold: float, version: str, fetch: Fetcher):
        self.booster, self.threshold, self.version, self.fetch = booster, threshold, str(version), fetch
        self.names = booster.feature_name()
        info = booster.dump_model(num_iteration=1)["feature_infos"]
        # categorical columns list their levels; LightGBM leaves constant (unused) columns out of feature_infos
        cat_names = [n for n in self.names if info.get(n, {}).get("values")]
        levels = booster.pandas_categorical or []
        if len(cat_names) != len(levels):
            raise ValueError(f"{len(cat_names)} categorical features but {len(levels)} category lists")
        self.cat_index = {n: {v: i for i, v in enumerate(lv)} for n, lv in zip(cat_names, levels)}
        self.col_pos = {n: j for j, n in enumerate(self.names)}

    # -- encoding -------------------------------------------------------------------------------
    def vectorise(self, txn: dict, feats: dict) -> np.ndarray:
        row = np.full((1, len(self.names)), np.nan)
        for src in (txn, feats):
            for name, v in src.items():
                j = self.col_pos.get(name)
                if j is None or v is None:
                    continue
                cats = self.cat_index.get(name)
                if cats is not None:
                    code = cats.get(v)  # unseen category -> missing, same as LightGBM's pandas path
                    if code is not None:
                        row[0, j] = code
                else:
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        continue
                    if not math.isnan(f):
                        row[0, j] = f
        return row

    # -- scoring --------------------------------------------------------------------------------
    def online_features(self, txn: dict):
        keys = {p: k for p, k in entity_keys(txn).items() if k is not None}
        return self.fetch(keys)

    def score(self, txn: dict, given: tuple[dict, list] | None = None) -> dict:
        """Score one transaction. `given` = (features, missing_entities) already fetched for it, which is how
        the shadow model scores the same inputs the champion saw without a second feature-store call."""
        t0 = time.perf_counter()
        feats, missing = given if given is not None else self.online_features(txn)
        t1 = time.perf_counter()
        row = self.vectorise(txn, feats)
        t2 = time.perf_counter()
        p = float(self.booster.predict(row, num_threads=1)[0])
        t3 = time.perf_counter()
        return {
            "score": p, "flagged": p >= self.threshold, "threshold": self.threshold,
            "model_version": self.version, "missing_entities": missing, "features": feats,
            "timings_ms": {"features": (t1 - t0) * 1e3, "encode": (t2 - t1) * 1e3,
                           "predict": (t3 - t2) * 1e3},
        }

    def explain_row(self, txn: dict) -> np.ndarray:
        """Encoded model input for one transaction (online features included), ready for SHAP."""
        feats, _ = self.online_features(txn)
        return self.vectorise(txn, feats)

    def explain(self, txn: dict, k: int = 3) -> list[dict]:
        """Top-k SHAP reasons in this process. ~0.36 s per row on the 1,490-tree model, so never on /score;
        the API runs this through explain_worker in separate processes instead."""
        return top_reasons_array(self.booster, self.explain_row(txn), self.names, k)


def monitored_columns(booster: lgb.Booster, n: int = 20) -> list[str]:
    """Columns echoed into scored-events and checked for drift: the model's n highest-gain features, the
    12 velocity features and the amount. The serving app and the drift monitor must agree on this list."""
    names = booster.feature_name()
    gain = booster.feature_importance("gain")
    top = [names[i] for i in np.argsort(gain)[::-1][:n]]
    velocity = [c for cols in VIEW_COLUMNS.values() for c in cols]
    return list(dict.fromkeys(top + velocity + ["TransactionAmt"]))
