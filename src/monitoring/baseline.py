"""Calibrate the drift alert: how much "drift" does Evidently report when nothing has changed?

Evidently's default per-column test (distance > 0.1) is too sensitive for short windows: a healthy
stream compared with a reference spread over weeks already shows many "drifted" columns (time of day,
sampling noise). This script
  1. cuts the validation split (the reference's own period: no real shift) into non-overlapping windows,
  2. uses every other window to learn a per-column threshold = the largest distance seen on a healthy
     window,
  3. checks those thresholds on the held-out validation windows (false alarms) and on replay-split
     windows, the newest data, where the model's PR-AUC really did fall (detection).

    python -m src.monitoring.baseline [window] [n_windows_per_group]
Writes reports/drift_thresholds_w<window>.json, which src.monitoring.drift loads.
"""
import json
import sys

import numpy as np
import pandas as pd

from src.data import PROCESSED
from src.features import add_velocity
from src.monitoring.drift import REPORTS, build_reference, evaluate, scored_frame
from src.serving.registry import load_alias
from src.serving.scorer import monitored_columns


def windows(df: pd.DataFrame, size: int, n: int) -> list[pd.DataFrame]:
    starts = np.linspace(0, len(df) - size, n).astype(int)  # evenly spread, non-overlapping when n*size<=len
    return [df.iloc[s:s + size].reset_index(drop=True) for s in starts]


def distances(ref, wins):
    return pd.DataFrame([evaluate(ref, w, thresholds=None)["distances"] for w in wins])


def main(size: int = 1000, n: int = 20):
    booster, _, version = load_alias("champion")
    cols = monitored_columns(booster)
    ref = build_reference(booster, cols)
    frames = {name: scored_frame(booster, add_velocity(pd.read_parquet(PROCESSED / f"{name}.parquet"))
                                 .reset_index(drop=True), cols) for name in ("val", "replay")}
    val_w = windows(frames["val"], size, 2 * n)
    calib, held = distances(ref, val_w[0::2]), distances(ref, val_w[1::2])
    replay = distances(ref, windows(frames["replay"], size, n))
    thr = calib.max()  # largest distance on any healthy calibration window, per column

    def share(d):
        return (d > thr).mean(axis=1)  # share of columns above their own calibrated threshold

    s_held, s_replay = share(held), share(replay)
    default = lambda d: (d > 0.1).mean(axis=1)
    out = {
        "window": size, "n_windows_per_group": n, "model_version": version,
        "thresholds": {c: float(v) for c, v in thr.items()},
        "calibrated": {"healthy_heldout": s_held.round(3).tolist(), "replay": s_replay.round(3).tolist(),
                       "healthy_mean": float(s_held.mean()), "healthy_max": float(s_held.max()),
                       "replay_mean": float(s_replay.mean()), "replay_min": float(s_replay.min())},
        "default_0.1": {"healthy_mean": float(default(held).mean()), "healthy_max": float(default(held).max()),
                        "replay_mean": float(default(replay).mean())},
        "score": {"healthy_distance_mean": float(held["score"].mean()),
                  "healthy_distance_max": float(held["score"].max()),
                  "replay_distance_mean": float(replay["score"].mean()),
                  "replay_windows_over_threshold": int((replay["score"] > thr["score"]).sum())},
        "replay_columns_over_threshold": {c: int((replay[c] > thr[c]).sum()) for c in thr.index
                                          if (replay[c] > thr[c]).any()},
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"drift_thresholds_w{size}.json").write_text(json.dumps(out, indent=2))
    c, d = out["calibrated"], out["default_0.1"]
    print(f"window {size}, {n} windows per group")
    print(f"default 0.1 test : healthy mean {d['healthy_mean']:.0%} (max {d['healthy_max']:.0%}), "
          f"replay mean {d['replay_mean']:.0%}")
    print(f"calibrated       : healthy mean {c['healthy_mean']:.0%} (max {c['healthy_max']:.0%}), "
          f"replay mean {c['replay_mean']:.0%} (min {c['replay_min']:.0%})")
    print(f"score distance   : healthy mean {out['score']['healthy_distance_mean']:.3f}, "
          f"replay mean {out['score']['replay_distance_mean']:.3f}; replay windows over threshold "
          f"{out['score']['replay_windows_over_threshold']}/{n}")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
