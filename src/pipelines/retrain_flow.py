"""Prefect flows: drift-triggered retraining and shadow-based promotion.

    python -m src.pipelines.retrain_flow retrain [--force] [--recent-weight 1]
    python -m src.pipelines.retrain_flow promote [--dry-run]

retrain   reads the drift monitor's log; if the last K evaluations all alerted, fits a challenger on everything
          labelled so far (same hyperparameters as the champion, fresh data, drifted rows included), picks
          its threshold on the newest labelled slice and registers it under the `challenger` alias.
promote   reads the shadow scores the API logged while the challenger ran, joins labels, applies the
          promotion rule and, only if it passes, moves the `champion` alias and tells the API to reload.
Every promote run appends its full comparison and verdict to reports/promotions.jsonl.
"""
import argparse
import json
import time
from pathlib import Path

import httpx
import lightgbm as lgb
import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task

from src.data import PROCESSED, TARGET
from src.features import ENTITIES, add_keys, velocity
from src.monitoring.drift import REPORTS
from src.pipelines import scenario
from src.pipelines.drift_injection import DriftSpec, apply_drift, top_numeric_features
from src.pipelines.promotion import PromotionRule, decide
from src.pipelines.shadow_eval import compare, join_labels, read_events
from src.serving.registry import MODEL_NAME, _client, load_alias, promote, register
from src.training.cost_matrix import best_threshold
from src.training.train import DROP

DRIFT_LOG = REPORTS / "drift_log.jsonl"
PROMOTION_LOG = REPORTS / "promotions.jsonl"


# ---- pure helpers (unit-tested) ---------------------------------------------------------------------
def should_retrain(records: list[dict], k: int = 3, model_version: str | None = None) -> bool:
    """True if the last `k` drift evaluations (of the serving model) all raised the dataset-drift alert.
    Requiring several in a row keeps one noisy window from starting a retrain."""
    recs = [r for r in records if model_version is None or str(r.get("model_version")) == str(model_version)]
    return len(recs) >= k and all(r["dataset_drifted"] for r in recs[-k:])


def with_velocity(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw transactions + offline velocity features computed on exactly these rows (past-only, as in
    src.features). Needed because drifted amounts change the rolling sums, so the precomputed parquet
    features no longer apply."""
    df = add_keys(raw)
    for prefix, key in ENTITIES.items():
        f = velocity(df, key, prefix).drop(columns=[key, "event_timestamp"])
        df = df.merge(f, on="TransactionID", how="left")
    return df.drop(columns=["event_timestamp", *ENTITIES.values()])


def champion_recipe(alias: str = "champion") -> tuple[dict, int]:
    """Hyperparameters and tree count the champion was trained with (logged to its MLflow run)."""
    mv = _client().get_model_version_by_alias(MODEL_NAME, alias)
    run = _client().get_run(mv.run_id)
    cast = lambda s: int(s) if s.lstrip("-").isdigit() else float(s)
    params = {k: cast(v) for k, v in run.data.params.items() if k not in ("threshold", "n_iter")}
    return params, int(run.data.metrics["n_iter"])


# ---- tasks ------------------------------------------------------------------------------------------
@task
def check_trigger(drift_log: str, k: int, force: bool) -> bool:
    log = get_run_logger()
    _, _, version = load_alias("champion")
    path = Path(drift_log)
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    fired = should_retrain(records, k, version)
    log.info("champion v%s: %d drift evaluations, retrain trigger %s%s", version, len(records),
             "FIRED" if fired else "not fired", " (forced)" if force and not fired else "")
    return fired or force


@task
def build_frames(drift_start: int, cutoff: int, spec: DriftSpec, thr_frac: float, recent_weight: float):
    """(fit, threshold-slice) frames. Fit = train + val + labelled replay before the threshold slice."""
    replay = pd.read_parquet(PROCESSED / "replay.parquet").reset_index(drop=True).iloc[:cutoff]
    replay = pd.concat([replay.iloc[:drift_start], apply_drift(replay.iloc[drift_start:], spec)])
    parts = [pd.read_parquet(PROCESSED / f"{n}.parquet") for n in ("train", "val")] + [replay]
    df = with_velocity(pd.concat(parts, ignore_index=True))
    n_old = len(df) - len(replay)
    cut_fit = cutoff - int((cutoff - drift_start) * thr_frac)
    drifted = np.arange(len(df)) >= n_old + drift_start
    weight = np.where(drifted, recent_weight, 1.0)
    split = n_old + cut_fit
    return df.iloc[:split], weight[:split], df.iloc[split:]


@task
def fit_challenger(fit: pd.DataFrame, weight: np.ndarray, thr: pd.DataFrame, force_label: str) -> dict:
    log = get_run_logger()
    champ, champ_thr, version = load_alias("champion")
    names = champ.feature_name()
    params, n_iter = champion_recipe()

    allX = pd.concat([fit.drop(columns=DROP)[names], thr.drop(columns=DROP)[names]])
    for c in allX.select_dtypes("object"):  # convert after concat: one shared category list for both frames
        allX[c] = allX[c].astype("category")
    # LightGBM casts DataFrames to float32 itself; doing it here halves peak memory
    allX = allX.astype({c: "float32" for c in allX.select_dtypes("float64").columns})
    Xfit, Xthr = allX.iloc[:len(fit)], allX.iloc[len(fit):]
    yfit, ythr = fit[TARGET], thr[TARGET]
    log.info("fitting on %d rows (%d frauds), threshold slice %d rows (%d frauds), %d trees", len(fit),
             yfit.sum(), len(thr), ythr.sum(), n_iter)
    t0 = time.time()
    p = dict(objective="binary", verbosity=-1, seed=42, learning_rate=0.05, bagging_freq=1, **params)
    booster = lgb.train(p, lgb.Dataset(Xfit, yfit, weight=weight), n_iter)
    amount = thr["TransactionAmt"].to_numpy()
    threshold, cost = best_threshold(ythr, booster.predict(Xthr), amount)
    # what simply re-tuning the champion's threshold on the same slice would have given (for the README)
    Xc = Xthr.copy()
    for c, levels in zip([c for c in names if str(Xc[c].dtype) == "category"], champ.pandas_categorical):
        Xc[c] = pd.Categorical(Xc[c].astype(object), categories=levels)
    champ_retuned, champ_retuned_cost = best_threshold(ythr, champ.predict(Xc[names]), amount)
    tag = time.strftime("%Y%m%d_%H%M%S")
    model_file, meta_file = PROCESSED / f"model_retrain_{tag}.txt", PROCESSED / f"model_retrain_{tag}.json"
    booster.save_model(str(model_file))  # not write_text: mangles newlines on Windows
    meta = {"threshold": threshold, "params": params, "n_iter": n_iter, "trained_from": f"champion v{version}",
            "fit_rows": len(fit), "threshold_slice_cost": cost,
            "champion_retuned_threshold": champ_retuned, "champion_retuned_slice_cost": champ_retuned_cost}
    meta_file.write_text(json.dumps(meta))
    new_version = register(str(model_file), str(meta_file), "challenger", f"retrain_{tag}{force_label}")
    _client().set_model_version_tag(MODEL_NAME, new_version, "champion_retuned_threshold", repr(champ_retuned))
    log.info("challenger v%s registered: threshold %.3g (champion's %.3g, re-tuned %.3g), %.0fs",
             new_version, threshold, champ_thr, champ_retuned, time.time() - t0)
    return {"version": new_version, **meta}


# ---- flows ------------------------------------------------------------------------------------------
@flow(name="retrain-challenger", log_prints=True)
def retrain_flow(drift_log: str = str(DRIFT_LOG), consecutive: int = 3, force: bool = False,
                 drift_start: int = scenario.DRIFT_START, cutoff: int = scenario.LABEL_CUTOFF,
                 thr_frac: float = scenario.THRESHOLD_FRAC, recent_weight: float = 1.0,
                 amount_scale: float = DriftSpec.amount_scale, feature_scale: float = DriftSpec.feature_scale):
    if not check_trigger(drift_log, consecutive, force):
        print("no retrain: drift trigger not fired")
        return None
    champ, _, _ = load_alias("champion")
    spec = DriftSpec(amount_scale, feature_scale, top_numeric_features(champ))
    fit, weight, thr = build_frames(drift_start, cutoff, spec, thr_frac, recent_weight)
    return fit_challenger(fit, weight, thr, "_forced" if force else "")


@task
def reload_api(url: str) -> dict:
    r = httpx.post(f"{url}/reload", timeout=120)
    r.raise_for_status()
    return r.json()


@flow(name="evaluate-and-promote", log_prints=True)
def promote_flow(first_event: int = scenario.LABEL_CUTOFF, last_event: int = scenario.SHADOW_END,
                 api_url: str = "http://localhost:8000", bootstrap: str = "localhost:19092",
                 dry_run: bool = False, rule: PromotionRule = PromotionRule()):
    mv = _client().get_model_version_by_alias(MODEL_NAME, "challenger")
    champ_version = _client().get_model_version_by_alias(MODEL_NAME, "champion").version
    if mv.version == champ_version:
        print("challenger is already the champion: nothing to evaluate")
        return None
    events = read_events(bootstrap, mv.version)
    ids = pd.read_parquet(PROCESSED / "replay.parquet", columns=["TransactionID"]).TransactionID
    window = set(ids.iloc[first_event:last_event])
    df = join_labels(events[events.TransactionID.isin(window)])
    if df.empty:
        print(f"no shadow events from challenger v{mv.version} in replay[{first_event}:{last_event}]")
        return None
    retuned = mv.tags.get("champion_retuned_threshold")
    cmp = compare(df, retuned_threshold=float(retuned) if retuned else None)
    verdict = decide(cmp, rule)
    print(f"challenger v{mv.version} vs champion v{champ_version}: {cmp['n_events']} events, "
          f"{cmp['n_frauds']} frauds, cost {cmp['champion']['expected_cost']:,.0f} -> "
          f"{cmp['challenger']['expected_cost']:,.0f} (saving {cmp['cost_saving']:.1%}, "
          f"CI {cmp['cost_saving_ci'][0]:.1%}..{cmp['cost_saving_ci'][1]:.1%})")
    if "champion_retuned" in cmp:
        print(f"for reference, champion with only its threshold re-tuned: cost "
              f"{cmp['champion_retuned']['expected_cost']:,.0f} ({cmp['champion_retuned']['saving_vs_champion']:.1%} saving)")
    print("PROMOTE" if verdict["promote"] else "KEEP CHAMPION: " + "; ".join(verdict["reasons"]))
    record = {"ts": time.time(), "challenger": mv.version, "champion": champ_version, "dry_run": dry_run,
              "window": [first_event, last_event], "comparison": cmp, **verdict}
    if verdict["promote"] and not dry_run:
        promote()
        record["reload"] = reload_api(api_url)
        print(f"champion alias -> v{mv.version}; API reloaded: {record['reload']}")
    REPORTS.mkdir(exist_ok=True)
    with open(PROMOTION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrain")
    r.add_argument("--force", action="store_true", help="retrain even if the drift trigger has not fired")
    r.add_argument("--consecutive", type=int, default=3)
    r.add_argument("--recent-weight", type=float, default=1.0)
    p = sub.add_parser("promote")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "retrain":
        retrain_flow(consecutive=a.consecutive, force=a.force, recent_weight=a.recent_weight)
    else:
        promote_flow(dry_run=a.dry_run)


if __name__ == "__main__":
    main()
