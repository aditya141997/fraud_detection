"""Drift monitor tests on synthetic data (Evidently is real; no Kafka or model needed)."""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import SHARE_THRESHOLD, evaluate


def frame(rng, n, amt_mean=4.0, product_p=(0.7, 0.2, 0.1)):
    return pd.DataFrame({
        "TransactionAmt": rng.lognormal(amt_mean, 0.8, n),
        "C13": rng.poisson(3, n).astype(float),
        "ProductCD": rng.choice(["W", "C", "R"], n, p=product_p),
        "score": rng.beta(1, 40, n),
    })


def test_no_shift_is_not_flagged():
    rng = np.random.default_rng(0)
    r = evaluate(frame(rng, 5000), frame(rng, 1000))
    assert r["drifted"] == [] and not r["dataset_drifted"]


def test_shifted_inputs_and_scores_are_flagged():
    rng = np.random.default_rng(1)
    cur = frame(rng, 1000, amt_mean=5.5, product_p=(0.2, 0.7, 0.1))
    cur["score"] = rng.beta(1, 5, 1000)
    r = evaluate(frame(rng, 5000), cur)
    assert {"TransactionAmt", "ProductCD", "score"} <= set(r["drifted"])
    assert r["score_drifted"] and r["dataset_drifted"] and r["drift_share"] > SHARE_THRESHOLD


def test_calibrated_thresholds_override_the_default():
    rng = np.random.default_rng(2)
    ref, cur = frame(rng, 5000), frame(rng, 1000, amt_mean=4.3)
    loose = evaluate(ref, cur, thresholds={c: 10.0 for c in cur.columns})
    tight = evaluate(ref, cur, thresholds={c: 0.0 for c in cur.columns})
    assert loose["drifted"] == [] and len(tight["drifted"]) == 4
    assert tight["calibrated"] and tight["share_threshold"] == SHARE_THRESHOLD


def test_all_missing_column_is_skipped_not_fatal():
    rng = np.random.default_rng(3)
    cur = frame(rng, 1000)
    cur["C13"] = np.nan
    r = evaluate(frame(rng, 5000), cur)
    assert "C13" not in r["distances"]
