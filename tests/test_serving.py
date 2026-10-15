"""Scoring-service tests. A tiny synthetic LightGBM model and a fake feature store: no Redis/Kafka/MLflow."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.serving.app as appmod
from src.serving.adapters import NullPublisher
from src.serving.scorer import VIEW_COLUMNS, Scorer

import lightgbm as lgb

PRODUCTS = ["W", "C", "R"]
VEL = [c for cols in VIEW_COLUMNS.values() for c in cols]


@pytest.fixture(scope="module")
def booster_and_frame():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({
        "TransactionAmt": rng.gamma(2, 50, n),
        "ProductCD": rng.choice(PRODUCTS, n),
        "card_txn_count_1h": rng.poisson(1, n).astype(float),
        "card_amt_sum_24h": rng.gamma(2, 80, n),
    })
    y = ((df.card_txn_count_1h > 2) | ((df.ProductCD == "C") & (df.TransactionAmt > 150))).astype(int)
    df["ProductCD"] = df["ProductCD"].astype("category")
    b = lgb.train({"objective": "binary", "verbosity": -1, "num_leaves": 8, "seed": 1},
                  lgb.Dataset(df, y), 40)
    return b, df


def make_scorer(booster, feats=None, missing=None, threshold=0.5):
    def fetch(keys):
        if feats is None:
            return {}, []
        return feats, missing or []
    return Scorer(booster, threshold, "7", fetch)


def test_numpy_path_matches_pandas_path(booster_and_frame):
    b, df = booster_and_frame
    s = make_scorer(b)
    for i in range(0, 300, 7):
        r = df.iloc[i]
        txn = {"TransactionAmt": float(r.TransactionAmt), "ProductCD": str(r.ProductCD)}
        feats = {"card_txn_count_1h": r.card_txn_count_1h, "card_amt_sum_24h": r.card_amt_sum_24h}
        got = float(b.predict(s.vectorise(txn, feats))[0])
        assert got == pytest.approx(float(b.predict(df.iloc[[i]])[0]), abs=1e-12)


def test_unseen_category_and_none_are_missing(booster_and_frame):
    b, _ = booster_and_frame
    s = make_scorer(b)
    row = s.vectorise({"TransactionAmt": 10.0, "ProductCD": "ZZ", "card1": 5}, {"card_txn_count_1h": None})
    assert np.isnan(row[0, s.col_pos["ProductCD"]])
    assert np.isnan(row[0, s.col_pos["card_txn_count_1h"]])
    assert row[0, s.col_pos["TransactionAmt"]] == 10.0


def test_threshold_decides_flag(booster_and_frame):
    b, _ = booster_and_frame
    txn = {"TransactionID": 1, "TransactionDT": 5, "TransactionAmt": 200.0, "ProductCD": "C"}
    feats = {"card_txn_count_1h": 5.0, "card_amt_sum_24h": 900.0}
    hi = make_scorer(b, feats, threshold=0.0).score(txn)
    lo = make_scorer(b, feats, threshold=1.1).score(txn)
    assert hi["flagged"] and not lo["flagged"] and hi["score"] == lo["score"]
    assert set(hi["timings_ms"]) == {"features", "encode", "predict"}


def test_explain_returns_positive_contributions_only(booster_and_frame):
    b, _ = booster_and_frame
    s = make_scorer(b, {"card_txn_count_1h": 6.0, "card_amt_sum_24h": 900.0})
    reasons = s.explain({"TransactionAmt": 200.0, "ProductCD": "C"}, k=3)
    assert 1 <= len(reasons) <= 3
    assert all(r["shap"] > 0 for r in reasons)
    assert reasons == sorted(reasons, key=lambda r: -r["shap"])


@pytest.fixture
def client(booster_and_frame, monkeypatch):
    b, _ = booster_and_frame
    state = {"feats": {"card_txn_count_1h": 5.0, "card_amt_sum_24h": 900.0}, "missing": [], "fail": False}

    def fetch(keys):
        if state["fail"]:
            raise ConnectionError("redis down")
        return state["feats"], state["missing"]

    monkeypatch.setattr(appmod, "build_scorer", lambda: Scorer(b, 0.5, "3", fetch))
    monkeypatch.setattr(appmod, "build_publisher", lambda scorer: NullPublisher())
    monkeypatch.setattr(appmod, "EXPLAIN_EAGER", False)  # the SHAP pool is built lazily in tests
    monkeypatch.setattr(appmod, "ALIAS_POLL_S", 0)  # no registry polling in tests
    with TestClient(appmod.app) as c:
        c.state = state
        yield c


TXN = {"TransactionID": 99, "TransactionDT": 1000, "TransactionAmt": 200.0, "ProductCD": "C",
       "card1": 1234, "card2": 111.0, "card3": 150.0, "card5": 226.0}


def test_score_endpoint(client):
    r = client.post("/score", json=TXN)
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == 99 and body["model_version"] == "3"
    assert 0 <= body["score"] <= 1 and body["flagged"] == (body["score"] >= 0.5)


def test_score_reports_missing_entities(client):
    client.state["missing"] = ["card"]
    assert client.post("/score", json=TXN).json()["missing_entities"] == ["card"]


def test_bad_input_rejected(client):
    assert client.post("/score", json={**TXN, "TransactionAmt": -5}).status_code == 422
    assert client.post("/score", json={"TransactionID": 1}).status_code == 422


def test_feature_store_outage_is_503_not_a_guess(client):
    client.state["fail"] = True
    r = client.post("/score", json=TXN)
    assert r.status_code == 503 and "ConnectionError" in r.json()["detail"]


def test_metrics_and_health(client):
    client.post("/score", json=TXN)
    m = client.get("/metrics").text
    assert "fraud_score_request_seconds_count" in m and "fraud_decisions_total" in m
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"] == "3"


def test_explain_endpoint(client):
    r = client.post("/explain", json=TXN)
    assert r.status_code == 200 and r.json()["transaction_id"] == 99
