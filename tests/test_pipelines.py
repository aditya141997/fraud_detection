"""Week 4 tests: retrain trigger, promotion rule, shadow comparison, drift injection, shadow serving.
Synthetic data only: no Redis, Kafka, MLflow server or dataset files."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.serving.app as appmod
from src.pipelines.drift_injection import DriftSpec, apply_drift, top_numeric_features
from src.pipelines.promotion import PromotionRule, decide
from src.pipelines.retrain_flow import should_retrain, with_velocity
from src.pipelines.shadow_eval import compare
from src.serving.scorer import Scorer


# ---- retrain trigger ---------------------------------------------------------------------------------
def recs(*flags, version="3"):
    return [{"dataset_drifted": f, "model_version": version} for f in flags]


def test_trigger_needs_consecutive_alerts():
    assert should_retrain(recs(False, True, True, True), k=3)
    assert not should_retrain(recs(True, True, False), k=3)       # latest window is healthy again
    assert not should_retrain(recs(True, True), k=3)              # not enough evaluations yet
    assert not should_retrain([], k=3)


def test_trigger_ignores_evaluations_of_other_model_versions():
    old = recs(True, True, True, version="2")
    assert not should_retrain(old + recs(True, version="3"), k=3, model_version="3")
    assert should_retrain(old + recs(True, True, True, version="3"), k=3, model_version="3")


# ---- shadow comparison + promotion rule --------------------------------------------------------------
def shadow_frame(n=8000, challenger_better=True, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.04).astype(int)
    amt = rng.gamma(2, 60, n) + 5
    champ = np.clip(0.02 + 0.10 * y + rng.normal(0, 0.05, n), 0, 1)          # weak ranking
    good = np.clip(0.02 + (0.45 if challenger_better else 0.10) * y + rng.normal(0, 0.05, n), 0, 1)
    return pd.DataFrame({"y": y, "amount": amt, "champ_score": champ, "champ_flag": champ >= 0.12,
                         "shadow_score": good, "shadow_flag": good >= 0.12})


def test_compare_reports_a_clear_gain_with_positive_interval():
    c = compare(shadow_frame(challenger_better=True), n_boot=300)
    assert c["cost_saving"] > 0.2 and c["cost_saving_ci"][0] > 0
    assert c["challenger"]["pr_auc"] > c["champion"]["pr_auc"]


def test_compare_identical_models_is_not_a_significant_gain():
    df = shadow_frame()
    df["shadow_score"], df["shadow_flag"] = df["champ_score"], df["champ_flag"]
    c = compare(df, n_boot=300)
    assert c["cost_saving"] == 0 and c["cost_saving_ci"] == [0.0, 0.0]
    assert not decide(c)["promote"]


def test_decide_promotes_a_clearly_better_challenger():
    c = compare(shadow_frame(challenger_better=True), n_boot=300)
    v = decide(c)
    assert v["promote"] and v["reasons"] == []


def good_cmp(**over):
    c = {"n_events": 10_000, "n_frauds": 400, "cost_saving": 0.10, "cost_saving_ci": [0.04, 0.16],
         "champion": {"pr_auc": 0.50, "flag_rate": 0.10}, "challenger": {"pr_auc": 0.55, "flag_rate": 0.11}}
    c.update(over)
    return c


@pytest.mark.parametrize("change, fragment", [
    ({"n_events": 1000}, "events"),
    ({"n_frauds": 20}, "frauds"),
    ({"cost_saving": 0.01}, "below"),
    ({"cost_saving_ci": [-0.01, 0.12]}, "not significant"),
    ({"challenger": {"pr_auc": 0.40, "flag_rate": 0.11}}, "PR-AUC"),
    ({"challenger": {"pr_auc": 0.55, "flag_rate": 0.30}}, "flag rate"),
    ({"challenger": {"pr_auc": 0.55, "flag_rate": 0.02}}, "flag rate"),
])
def test_each_check_can_block_promotion(change, fragment):
    v = decide(good_cmp(**change))
    assert not v["promote"] and any(fragment in r for r in v["reasons"])


def test_decide_passes_baseline_case_and_rule_is_configurable():
    assert decide(good_cmp())["promote"]
    assert not decide(good_cmp(), PromotionRule(min_saving=0.2))["promote"]


# ---- drift injection ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def booster():
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame({"TransactionAmt": rng.gamma(2, 50, n), "C13": rng.poisson(3, n).astype(float),
                       "C1": rng.poisson(2, n).astype(float), "noise": rng.normal(size=n),
                       "ProductCD": pd.Categorical(rng.choice(list("WCR"), n)),
                       "card_txn_count_1h": rng.poisson(1, n).astype(float)})
    y = ((df.C13 > 4) | (df.C1 > 4)).astype(int)
    return lgb.train({"objective": "binary", "verbosity": -1, "num_leaves": 8}, lgb.Dataset(df, y), 30)


def test_top_features_skip_amount_velocity_and_categoricals(booster):
    top = top_numeric_features(booster, k=3)
    assert set(top[:2]) == {"C13", "C1"}
    assert not {"TransactionAmt", "card_txn_count_1h", "ProductCD"} & set(top)


def test_apply_drift_scales_and_keeps_missing():
    df = pd.DataFrame({"TransactionAmt": [10.0, 20.0], "C13": [np.nan, 4.0], "other": [1.0, 2.0]})
    out = apply_drift(df, DriftSpec(amount_scale=3.0, feature_scale=2.0, features=("C13",)))
    assert out.TransactionAmt.tolist() == [30.0, 60.0]
    assert np.isnan(out.C13[0]) and out.C13[1] == 8.0
    assert out.other.tolist() == [1.0, 2.0] and df.TransactionAmt.tolist() == [10.0, 20.0]  # input untouched


# ---- offline velocity for retraining -----------------------------------------------------------------
def test_with_velocity_is_past_only_and_keeps_row_order():
    raw = pd.DataFrame({"TransactionID": [1, 2, 3, 4], "TransactionDT": [0, 600, 1200, 90_000],
                        "TransactionAmt": [10.0, 20.0, 30.0, 40.0], "card1": [7, 7, 7, 7], "card2": [1.0] * 4,
                        "card3": [2.0] * 4, "card5": [3.0] * 4, "DeviceInfo": [None] * 4,
                        "P_emaildomain": ["a.com"] * 4})
    out = with_velocity(raw)
    assert out.TransactionID.tolist() == [1, 2, 3, 4]
    assert out.card_txn_count_1h.tolist() == [0, 1, 2, 0]                 # current event excluded
    assert out.card_amt_sum_1h.tolist() == [0.0, 10.0, 30.0, 0.0]
    assert out.card_txn_count_24h.tolist() == [0, 1, 2, 0]                # event 4 is > 24h after the others
    assert out.device_txn_count_1h.isna().all()                           # no device key -> missing
    assert not {"event_timestamp", "card_id", "device_info", "email_domain"} & set(out.columns)


# ---- shadow mode in the API --------------------------------------------------------------------------
class Capture:
    dropped = 0

    def __init__(self):
        self.events = []

    def publish(self, txn, result):
        self.events.append(result)

    def close(self, timeout=0):
        pass


@pytest.fixture
def shadow_client(booster, monkeypatch):
    b2 = lgb.train({"objective": "binary", "verbosity": -1, "num_leaves": 4, "seed": 3}, lgb.Dataset(
        pd.DataFrame({"TransactionAmt": np.linspace(1, 300, 400), "C13": np.arange(400) % 9 * 1.0,
                      "C1": np.arange(400) % 5 * 1.0, "noise": np.zeros(400),
                      "ProductCD": pd.Categorical(["W", "C", "R", "W"] * 100),
                      "card_txn_count_1h": np.zeros(400)}), (np.arange(400) % 9 > 5).astype(int)), 10)
    cap = Capture()
    state = {"fail_shadow": False}

    fetch = lambda keys: ({"card_txn_count_1h": 1.0}, [])
    champion = Scorer(booster, 0.5, "3", fetch)
    shadow = Scorer(b2, 0.3, "4", fetch)
    real_score = shadow.score

    def maybe_fail(txn, given=None):
        if state["fail_shadow"]:
            raise RuntimeError("boom")
        return real_score(txn, given)

    shadow.score = maybe_fail
    monkeypatch.setattr(appmod, "build_scorer", lambda: champion)
    monkeypatch.setattr(appmod, "build_shadow", lambda c: shadow)
    monkeypatch.setattr(appmod, "build_publisher", lambda s: cap)
    monkeypatch.setattr(appmod, "EXPLAIN_EAGER", False)
    monkeypatch.setattr(appmod, "ALIAS_POLL_S", 0)
    with TestClient(appmod.app) as c:
        c.cap, c.state = cap, state
        yield c


TXN = {"TransactionID": 5, "TransactionDT": 100, "TransactionAmt": 120.0, "ProductCD": "C", "C13": 6.0, "C1": 1.0}


def test_shadow_scores_are_logged_but_never_returned(shadow_client):
    body = shadow_client.post("/score", json=TXN).json()
    assert body["model_version"] == "3"                      # the client only ever sees the champion
    assert "shadow" not in body and "shadow_score" not in body
    r = shadow_client.cap.events[-1]
    assert r["shadow"]["model_version"] == "4" and 0 <= r["shadow"]["score"] <= 1
    assert shadow_client.get("/health").json()["shadow_version"] == "4"
    assert "fraud_shadow_decisions_total" in shadow_client.get("/metrics").text


def test_shadow_failure_does_not_fail_the_request(shadow_client):
    shadow_client.state["fail_shadow"] = True
    r = shadow_client.post("/score", json=TXN)
    assert r.status_code == 200 and r.json()["model_version"] == "3"
    assert "shadow" not in shadow_client.cap.events[-1]
    assert "fraud_shadow_errors_total" in shadow_client.get("/metrics").text


# ---- shadow / alias bookkeeping ----------------------------------------------------------------------
def test_shadow_is_off_when_challenger_is_the_champion(booster, monkeypatch):
    """MLflow hands back alias versions as ints while Scorer stores strs; they must still compare equal."""
    monkeypatch.setattr(appmod, "SHADOW_ALIAS", "challenger")
    champion = Scorer(booster, 0.5, 5, lambda k: ({}, []))                      # Scorer stores "5"
    monkeypatch.setattr(appmod, "load_alias", lambda alias: (booster, 0.3, 5))  # int, like MLflow
    assert appmod.build_shadow(champion) is None
    monkeypatch.setattr(appmod, "load_alias", lambda alias: (booster, 0.3, 6))
    assert appmod.build_shadow(champion).version == "6"


def test_shadow_alias_missing_means_no_shadow_not_a_crash(booster, monkeypatch):
    monkeypatch.setattr(appmod, "SHADOW_ALIAS", "challenger")
    monkeypatch.setattr(appmod, "load_alias", lambda alias: (_ for _ in ()).throw(RuntimeError("no alias")))
    assert appmod.build_shadow(Scorer(booster, 0.5, 5, lambda k: ({}, []))) is None


@pytest.mark.parametrize("registry, serving, expect", [
    ({"champion": "5", "challenger": "5"}, ("5", None), False),   # promoted and already converged
    ({"champion": "5", "challenger": "5"}, ("3", "5"), True),     # promotion happened elsewhere: reload
    ({"champion": "3", "challenger": "6"}, ("3", "5"), True),     # new challenger registered
    ({"champion": "3", "challenger": "5"}, ("3", "5"), False),
    ({"champion": None, "challenger": "5"}, ("3", "5"), False),   # registry unavailable: keep serving
])
def test_alias_drift_detects_when_a_worker_is_behind(booster, monkeypatch, registry, serving, expect):
    monkeypatch.setattr(appmod, "SHADOW_ALIAS", "challenger")
    monkeypatch.setattr(appmod, "alias_version", lambda alias: registry[alias])
    champ_v, shadow_v = serving
    fetch = lambda k: ({}, [])
    appmod.app.state.scorer = Scorer(booster, 0.5, champ_v, fetch)
    appmod.app.state.shadow = Scorer(booster, 0.3, shadow_v, fetch) if shadow_v else None
    assert appmod._alias_drift() is expect
