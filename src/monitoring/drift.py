"""Drift monitor: compares a sliding window of live scored events with a validation-split reference.

Consumes the `scored-events` topic written by the scoring API, runs an Evidently data-drift report over
the monitored features plus the model score, exposes the result as Prometheus gauges (default :8001) and
writes the latest HTML report. Week 4's retrain flow triggers on these gauges.

    python -m src.monitoring.drift                 # follow the topic
    python -m src.monitoring.drift --once          # evaluate what is on the topic now, then exit

Evidently's per-column drift test here is its default: Wasserstein distance (normalised) for numeric
columns and Jensen-Shannon distance for categorical ones, each against a 0.1 threshold, so a column
counts as drifted when its distance exceeds 0.1.
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from prometheus_client import Gauge, start_http_server

from src.data import PROCESSED
from src.features import add_velocity
from src.serving.scorer import monitored_columns

TOPIC = "scored-events"
REPORTS = Path("reports")
SHARE_THRESHOLD = 0.25  # drifted-column share above which a window alerts (calibrated thresholds)
SHARE_THRESHOLD_UNCALIBRATED = 0.5  # fallback with Evidently's default 0.1 per-column threshold

G_SHARE = Gauge("fraud_drift_share", "Share of monitored columns with drift in the current window")
G_COLS = Gauge("fraud_drift_columns", "Number of drifted columns in the current window")
G_SCORE = Gauge("fraud_score_drift_distance", "Drift distance of the model score vs reference")
G_SCORE_FLAG = Gauge("fraud_score_drift_detected", "1 if the score distribution has drifted")
G_DRIFTED = Gauge("fraud_dataset_drift_detected", "1 if the drifted-column share exceeds the threshold")
G_WINDOW = Gauge("fraud_drift_window_events", "Events in the evaluated window")
G_LAST = Gauge("fraud_drift_last_run_timestamp", "Unix time of the last drift evaluation")
G_FEATURE = Gauge("fraud_feature_drift_distance", "Per-feature drift distance", ["feature"])


def scored_frame(booster: lgb.Booster, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Monitored columns + the model's offline score for rows that already carry velocity features."""
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    names = booster.feature_name()
    cats = [c for c in names if X[c].dtype == object]
    for c, levels in zip(cats, booster.pandas_categorical):
        X[c] = pd.Categorical(X[c], categories=levels)
    out = df[cols].copy().reset_index(drop=True)
    out["score"] = booster.predict(X[names])
    return out


def build_reference(booster: lgb.Booster, cols: list[str], n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Validation-split rows (features + the model's score on them): what 'normal' looked like at tuning."""
    val = add_velocity(pd.read_parquet(PROCESSED / "val.parquet")).sample(n, random_state=seed)
    return scored_frame(booster, val, cols)


def _frame(rows: list[dict], cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([{**r["features"], "score": r["score"]} for r in rows])
    return df.reindex(columns=cols + ["score"])


def load_thresholds(window: int) -> dict | None:
    """Per-column thresholds learned by src.monitoring.baseline for this window size, if available."""
    f = REPORTS / f"drift_thresholds_w{window}.json"
    return json.loads(f.read_text())["thresholds"] if f.exists() else None


def evaluate(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path | None = None,
             thresholds: dict | None = None) -> dict:
    """Run the Evidently drift preset. A column drifted if its distance exceeds its calibrated threshold
    (from `thresholds`) or, without one, Evidently's default 0.1."""
    cols = [c for c in reference.columns if reference[c].notna().any() and current[c].notna().any()]
    cat = [c for c in cols if reference[c].dtype == object or current[c].dtype == object]
    num = [c for c in cols if c not in cat]
    # make dtypes consistent between the two frames
    ref, cur = reference[cols].copy(), current[cols].copy()
    for c in num:
        ref[c], cur[c] = pd.to_numeric(ref[c], errors="coerce"), pd.to_numeric(cur[c], errors="coerce")
    for c in cat:
        ref[c], cur[c] = ref[c].astype("string").astype(object), cur[c].astype("string").astype(object)
    dd = DataDefinition(numerical_columns=num, categorical_columns=cat)
    snap = Report([DataDriftPreset()]).run(Dataset.from_pandas(cur, data_definition=dd),
                                           Dataset.from_pandas(ref, data_definition=dd))
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snap.save_html(str(html_path))
    distances, drifted = {}, []
    for m in snap.dict()["metrics"]:
        if m["metric_name"].startswith("ValueDrift"):
            col = m["config"]["column"]
            thr = (thresholds or {}).get(col, float(m["config"].get("threshold", 0.1)))
            distances[col] = float(m["value"])
            if distances[col] > thr:
                drifted.append(col)
    share = len(drifted) / max(len(distances), 1)
    limit = SHARE_THRESHOLD if thresholds else SHARE_THRESHOLD_UNCALIBRATED
    return {"drift_share": share, "n_columns": len(distances), "drifted": sorted(drifted),
            "distances": distances, "score_distance": distances.get("score"),
            "score_drifted": "score" in drifted, "share_threshold": limit,
            "dataset_drifted": share > limit, "calibrated": bool(thresholds)}


def publish(result: dict, window: int) -> None:
    G_SHARE.set(result["drift_share"] or 0)
    G_COLS.set(len(result["drifted"]))
    G_SCORE.set(result["score_distance"] or 0)
    G_SCORE_FLAG.set(int(result["score_drifted"]))
    G_DRIFTED.set(int(result["dataset_drifted"]))
    G_WINDOW.set(window)
    G_LAST.set(time.time())
    for c, d in result["distances"].items():
        G_FEATURE.labels(c).set(d)


def main():
    from confluent_kafka import Consumer
    from src.serving.registry import load_alias

    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:19092")
    ap.add_argument("--group", default="drift-monitor")
    ap.add_argument("--window", type=int, default=1000, help="most recent events compared with the reference")
    ap.add_argument("--min-events", type=int, default=500)
    ap.add_argument("--every", type=int, default=250, help="re-evaluate after this many new events")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--from-beginning", action="store_true")
    ap.add_argument("--once", action="store_true", help="evaluate once the topic is drained, then exit")
    a = ap.parse_args()

    booster, _, version = load_alias("champion")
    cols = monitored_columns(booster)
    ref = build_reference(booster, cols)
    print(f"reference: {len(ref):,} validation rows, {len(cols)} monitored columns + score "
          f"(champion v{version})", flush=True)
    thresholds = load_thresholds(a.window)
    print("per-column thresholds: " + (f"calibrated (window {a.window})" if thresholds else
          "none found, using Evidently default 0.1 (run python -m src.monitoring.baseline)"), flush=True)
    start_http_server(a.port)

    group = f"{a.group}-once-{int(time.time())}" if a.once else a.group  # --once must re-read from the start
    c = Consumer({"bootstrap.servers": a.bootstrap, "group.id": group,
                  "auto.offset.reset": "earliest" if (a.from_beginning or a.once) else "latest"})
    c.subscribe([TOPIC])
    window, since, idle = deque(maxlen=a.window), 0, 0
    REPORTS.mkdir(exist_ok=True)

    def run():
        res = evaluate(ref, _frame(list(window), cols), REPORTS / "drift_latest.html", thresholds)
        publish(res, len(window))
        with open(REPORTS / "drift_log.jsonl", "a") as f:
            f.write(json.dumps({"ts": time.time(), "window": len(window), "model_version": version, **res}) + "\n")
        print(f"[{len(window):,} events] drift share {res['drift_share']:.0%} "
              f"({len(res['drifted'])}/{res['n_columns']} columns, alert above {res['share_threshold']:.0%}); score distance "
              f"{res['score_distance']:.3f}; drifted: {res['drifted']}", flush=True)
        return res

    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                idle += 1
                if a.once and idle >= 3 and len(window) >= a.min_events:
                    break
                continue
            idle = 0
            if msg.error():
                raise RuntimeError(msg.error())
            window.append(json.loads(msg.value()))
            since += 1
            if len(window) >= a.min_events and since >= a.every:
                run()
                since = 0
        if since:
            run()
    finally:
        c.close()


if __name__ == "__main__":
    main()
