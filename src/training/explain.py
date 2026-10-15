"""Top-k SHAP reasons per prediction.

Uses LightGBM's native TreeSHAP (pred_contrib=True), which gives the same exact values as the
shap library's TreeExplainer without a second dependency on the serving path.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd


def top_reasons(booster: lgb.Booster, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """For each row, the k features pushing the score towards fraud the most."""
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]  # last column is the base value
    cols = np.array(X.columns)
    out = []
    for i in range(len(X)):
        idx = np.argsort(contrib[i])[::-1][:k]
        out.append([
            {"feature": cols[j], "shap": round(float(contrib[i, j]), 4), "value": _py(X.iloc[i, j])}
            for j in idx if contrib[i, j] > 0
        ])
    return out


def top_reasons_array(booster: lgb.Booster, row: np.ndarray, names: list[str], k: int = 3) -> list[dict]:
    """Same as top_reasons for one already-encoded row (1 x n_features float array, NaN = missing).

    Used by the serving path, which builds numpy rows directly instead of DataFrames. `value` is the
    encoded value, so a categorical feature shows its category code."""
    contrib = booster.predict(row, pred_contrib=True, num_threads=1)[0, :-1]
    return [
        {"feature": names[j], "shap": round(float(contrib[j]), 4), "value": _py(row[0, j])}
        for j in np.argsort(contrib)[::-1][:k] if contrib[j] > 0
    ]


def _py(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    return v.item() if hasattr(v, "item") else v
