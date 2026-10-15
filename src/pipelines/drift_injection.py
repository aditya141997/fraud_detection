"""Controlled, synthetic drift for the replay stream.

The shift is applied to the raw transaction before its velocity features are computed, so the online
(Redis) features, the scored payload and the offline retraining frame all see the same drifted world.
It is a covariate shift: the transaction amount and the model's most important numeric inputs are scaled.
Fraud labels are untouched, so how much the shift hurts the champion, and whether retraining helps, is
measured, not assumed (see README).
"""
from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd

from src.serving.scorer import VIEW_COLUMNS

VELOCITY = {c for cols in VIEW_COLUMNS.values() for c in cols}


@dataclass(frozen=True)
class DriftSpec:
    amount_scale: float = 2.5
    feature_scale: float = 2.0
    features: tuple[str, ...] = ()


def top_numeric_features(booster: lgb.Booster, k: int = 3) -> tuple[str, ...]:
    """The k highest-gain numeric, non-velocity inputs of the model (the columns whose shift matters most)."""
    names = booster.feature_name()
    info = booster.dump_model(num_iteration=1)["feature_infos"]
    gain = booster.feature_importance("gain")
    ranked = sorted(range(len(names)), key=lambda i: -gain[i])
    keep = [names[i] for i in ranked
            if names[i] not in VELOCITY and names[i] != "TransactionAmt"
            and not info.get(names[i], {}).get("values")]
    return tuple(keep[:k])


def apply_drift(df: pd.DataFrame, spec: DriftSpec) -> pd.DataFrame:
    """Shifted copy of raw transactions (missing values stay missing)."""
    out = df.copy()
    out["TransactionAmt"] = out["TransactionAmt"] * spec.amount_scale
    for c in spec.features:
        out[c] = out[c] * spec.feature_scale
    return out
