"""Experiments that close the README's "known limitations".

Same protocol as tune.py: fit on train, choose hyperparameters + threshold on validation, report on the
untouched replay split. Stages cache to data/processed/caveats_results.json so a run can resume.

    python -m src.training.caveats [stage ...]    stages: search ablation focal cost delay
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score

from src.data import PROCESSED
from src.training.cost_matrix import expected_cost, no_model_cost
import src.training.cost_matrix as cm
from src.training.train import recall_at_fpr, split_xy
from src.training.tune import AMT, align, load

OUT = PROCESSED / "caveats_results.json"
VELOCITY_PREFIXES = ("card_txn", "card_amt", "device_txn", "device_amt", "email_txn", "email_amt")
TRIALS = 30
MAX_TREES = 4000
BASE = dict(objective="binary", verbosity=-1, learning_rate=0.05, bagging_freq=1)
GRID = np.linspace(0.30, 0.9999, 700)  # wider than cost_matrix.best_threshold's 0.80-0.9995


def thr_wide(y, score, amount, review_cost=None):
    """Cost-minimising threshold over a wide quantile grid. Returns (threshold, cost, at_edge)."""
    old = cm.REVIEW_COST
    cm.REVIEW_COST = old if review_cost is None else review_cost
    try:
        grid = np.quantile(score, GRID)
        costs = np.array([expected_cost(y, score, amount, t) for t in grid])
    finally:
        cm.REVIEW_COST = old
    i = int(costs.argmin())
    return float(grid[i]), float(costs[i]), bool(i in (0, len(grid) - 1))


class Data:
    def __init__(self):
        train, val, replay = load("train"), load("val"), load("replay")
        self.train_time = train["TransactionDT"].to_numpy()
        self.Xtr, self.ytr = split_xy(train)
        cats = list(self.Xtr.select_dtypes("category").columns)
        self.Xva, self.yva = split_xy(val)
        self.Xre, self.yre = split_xy(replay)
        self.Xva, self.Xre = align(self.Xva, self.Xtr, cats), align(self.Xre, self.Xtr, cats)
        self.amt_va, self.amt_re = val[AMT].to_numpy(), replay[AMT].to_numpy()
        self.vel = [c for c in self.Xtr.columns if c.startswith(VELOCITY_PREFIXES)]
        self.val_start = val["TransactionDT"].min()


def fit(d, params, cols=None, rows=None, rounds=MAX_TREES, fobj=None, feval=None):
    cols = list(d.Xtr.columns) if cols is None else cols
    rows = np.ones(len(d.Xtr), dtype=bool) if rows is None else rows
    dtr = lgb.Dataset(d.Xtr.loc[rows, cols], d.ytr[rows], params={"feature_pre_filter": False})
    dva = lgb.Dataset(d.Xva[cols], d.yva, reference=dtr)
    p = dict(params)
    if fobj is not None:
        p["objective"] = fobj
        p["metric"] = "None"
    else:
        p["metric"] = "average_precision"
    b = lgb.train(p, dtr, rounds, valid_sets=[dva], feval=feval,
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return b


def score(b, X, cols, focal=False):
    raw = b.predict(X[cols], num_iteration=b.best_iteration, raw_score=focal)
    return expit(raw) if focal else raw


def summarise(d, va, re, review_cost=None):
    """Metrics on replay with threshold chosen on validation."""
    thr, _, edge = thr_wide(d.yva.to_numpy(), va, d.amt_va, review_cost)
    old = cm.REVIEW_COST
    cm.REVIEW_COST = old if review_cost is None else review_cost
    try:
        cost = expected_cost(d.yre.to_numpy(), re, d.amt_re, thr)
    finally:
        cm.REVIEW_COST = old
    return {
        "val_pr_auc": average_precision_score(d.yva, va),
        "pr_auc": average_precision_score(d.yre, re),
        "recall_at_1pct_fpr": recall_at_fpr(d.yre, re),
        "threshold": thr, "threshold_at_grid_edge": edge,
        "flag_rate": float((re >= thr).mean()), "cost": cost,
    }


def stage_search(d, res):
    """Wider Optuna search: the first search hit its tree cap and the edges of three parameters."""
    def objective(trial):
        p = dict(BASE, seed=42,
                 num_leaves=trial.suggest_int("num_leaves", 15, 511, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 300),
                 feature_fraction=trial.suggest_float("feature_fraction", 0.1, 1.0),
                 bagging_fraction=trial.suggest_float("bagging_fraction", 0.4, 1.0),
                 lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
                 scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True))
        b = fit(d, p)
        trial.set_user_attr("best_iter", b.best_iteration)
        print(f"  trial {trial.number}: pr_auc={b.best_score['valid_0']['average_precision']:.4f} "
              f"iters={b.best_iteration}", flush=True)
        return b.best_score["valid_0"]["average_precision"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=TRIALS)
    best, n_iter = study.best_params, study.best_trial.user_attrs["best_iter"]
    cols = list(d.Xtr.columns)
    b = fit(d, dict(BASE, seed=42, **best))
    r = summarise(d, score(b, d.Xva, cols), score(b, d.Xre, cols))
    r.update(params=best, n_iter=b.best_iteration, max_trees=MAX_TREES, trials=TRIALS,
             at_tree_cap=bool(b.best_iteration >= MAX_TREES - 1))
    res["search"] = r
    b.save_model(str(PROCESSED / "model_v2.txt"))
    PROCESSED.joinpath("model_v2.json").write_text(json.dumps(
        {"threshold": r["threshold"], "features": cols, "params": best, "n_iter": b.best_iteration}))


def stage_ablation(d, res):
    """Same tuned params, with and without the 12 velocity columns, 3 seeds each."""
    params = dict(BASE, **res["search"]["params"])
    all_cols = list(d.Xtr.columns)
    no_vel = [c for c in all_cols if c not in d.vel]
    out = {"velocity_columns": d.vel}
    for name, cols in (("with_velocity", all_cols), ("without_velocity", no_vel)):
        runs = []
        for seed in (1, 2, 3):
            b = fit(d, dict(params, seed=seed), cols=cols)
            runs.append(summarise(d, score(b, d.Xva, cols), score(b, d.Xre, cols)))
            print(f"  {name} seed={seed}: replay pr_auc={runs[-1]['pr_auc']:.4f} cost={runs[-1]['cost']:,.0f}",
                  flush=True)
        out[name] = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                     for k in ("val_pr_auc", "pr_auc", "recall_at_1pct_fpr", "cost")}
    res["ablation"] = out


def focal_obj(alpha, gamma):
    def grad(z, y):
        p = np.clip(expit(z), 1e-7, 1 - 1e-7)
        g1 = alpha * (1 - p) ** gamma * (gamma * p * np.log(p) - (1 - p))
        g0 = (1 - alpha) * p ** gamma * (-gamma * (1 - p) * np.log(1 - p) + p)
        return np.where(y == 1, g1, g0)

    def obj(z, data):
        y = data.get_label()
        g = grad(z, y)
        h = (grad(z + 1e-3, y) - grad(z - 1e-3, y)) / 2e-3
        return g, np.maximum(h, 1e-4)  # focal loss is non-convex; clip a negative Hessian
    return obj


def stage_focal(d, res):
    """Focal loss (custom objective) vs the weighted-binary model, same tuned tree params."""
    params = {k: v for k, v in res["search"]["params"].items() if k != "scale_pos_weight"}
    params = dict(BASE, seed=42, **params)
    cols = list(d.Xtr.columns)
    pr = lambda z, data: ("pr_auc", average_precision_score(data.get_label(), z), True)  # monotone in z
    out = []
    for gamma in (1.0, 2.0):
        for alpha in (0.5, 0.75, 0.9):
            b = fit(d, params, fobj=focal_obj(alpha, gamma), feval=pr)
            r = summarise(d, score(b, d.Xva, cols, True), score(b, d.Xre, cols, True))
            r.update(alpha=alpha, gamma=gamma, n_iter=b.best_iteration)
            out.append(r)
            print(f"  focal a={alpha} g={gamma}: val={r['val_pr_auc']:.4f} replay={r['pr_auc']:.4f} "
                  f"cost={r['cost']:,.0f}", flush=True)
    res["focal"] = {"runs": out, "best_by_val_pr_auc": max(out, key=lambda r: r["val_pr_auc"])}


def stage_cost(d, res):
    """How the threshold and the savings move with the assumed review cost."""
    cols = list(d.Xtr.columns)
    b = lgb.Booster(model_file=str(PROCESSED / "model_v2.txt"))
    va, re = b.predict(d.Xva[cols]), b.predict(d.Xre[cols])
    old, out = cm.REVIEW_COST, []
    for rc in (1.0, 5.0, 10.0, 25.0, 50.0, 100.0):
        r = summarise(d, va, re, rc)
        cm.REVIEW_COST = rc
        r["review_cost"], r["no_model_cost"] = rc, no_model_cost(d.yre.to_numpy(), d.amt_re)
        cm.REVIEW_COST = old
        r["saving_vs_no_model"] = 1 - r["cost"] / r["no_model_cost"]
        out.append({k: r[k] for k in ("review_cost", "threshold", "flag_rate", "cost", "no_model_cost",
                                      "saving_vs_no_model", "recall_at_1pct_fpr")})
        print(f"  review_cost={rc:>5}: thr={r['threshold']:.4f} flag={r['flag_rate']:.2%} "
              f"cost={r['cost']:,.0f} saving={r['saving_vs_no_model']:.1%}", flush=True)
    res["cost"] = out


def stage_delay(d, res):
    """Labels arrive late: a retrain at time T only knows labels for transactions older than T - delay."""
    params = dict(BASE, seed=42, **res["search"]["params"])
    cols = list(d.Xtr.columns)
    out = []
    for days in (0, 7, 14, 30):
        keep = d.train_time < d.val_start - days * 86400
        b = fit(d, params, rows=keep)
        r = summarise(d, score(b, d.Xva, cols), score(b, d.Xre, cols))
        r.update(delay_days=days, train_rows=int(keep.sum()), train_fraud=int(d.ytr[keep].sum()))
        out.append(r)
        print(f"  delay={days:>2}d rows={keep.sum():,}: replay pr_auc={r['pr_auc']:.4f} cost={r['cost']:,.0f}",
              flush=True)
    res["delay"] = out


STAGES = {"search": stage_search, "ablation": stage_ablation, "focal": stage_focal,
          "cost": stage_cost, "delay": stage_delay}


def main():
    wanted = sys.argv[1:] or list(STAGES)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    d = Data()
    res["no_model_cost_replay"] = no_model_cost(d.yre.to_numpy(), d.amt_re)
    for name in wanted:
        t = time.time()
        print(f"== {name}", flush=True)
        STAGES[name](d, res)
        OUT.write_text(json.dumps(res, indent=2, default=float))
        print(f"== {name} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
