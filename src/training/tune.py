"""Week 2: Optuna tuning with velocity features, cost-based threshold, final report on the replay split.

Protocol: fit on train, tune hyperparameters + choose threshold on validation, report once on the
untouched replay split (the newest 15%). The baseline gets the same treatment.
"""
import json
import sys

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import PROCESSED, TARGET
from src.features import add_velocity
from src.training.cost_matrix import best_threshold, expected_cost, no_model_cost
from src.training.train import recall_at_fpr, split_xy

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
AMT = "TransactionAmt"


def load(name):
    return add_velocity(pd.read_parquet(PROCESSED / f"{name}.parquet"))


def align(X, ref, cats):
    for c in cats:
        X[c] = pd.Categorical(X[c], categories=ref[c].cat.categories)
    return X


def evaluate(name, y, score, amount, thr):
    m = {
        "pr_auc": average_precision_score(y, score),
        "recall_at_1pct_fpr": recall_at_fpr(y, score),
        "expected_cost": expected_cost(y.to_numpy(), score, amount.to_numpy(), thr),
    }
    print(f"{name}: " + ", ".join(f"{k}={v:,.4f}" for k, v in m.items()))
    return m


def main():
    mlflow.set_experiment("fraud-week2")
    train, val, replay = load("train"), load("val"), load("replay")
    Xtr, ytr = split_xy(train)
    cats = list(Xtr.select_dtypes("category").columns)
    Xva, yva = split_xy(val)
    Xre, yre = split_xy(replay)
    Xva, Xre = align(Xva, Xtr, cats), align(Xre, Xtr, cats)
    amt_va, amt_re = val[AMT], replay[AMT]
    print(f"no-model cost on replay: {no_model_cost(yre.to_numpy(), amt_re.to_numpy()):,.0f}")

    # --- baseline: logistic regression (numeric columns only), threshold chosen on validation
    num = [c for c in Xtr.columns if c not in cats]
    lr = make_lr(num).fit(Xtr, ytr)
    thr_lr, _ = best_threshold(yva, lr.predict_proba(Xva)[:, 1], amt_va)
    with mlflow.start_run(run_name="logreg_baseline"):
        m_lr = evaluate("logreg@replay", yre, lr.predict_proba(Xre)[:, 1], amt_re, thr_lr)
        mlflow.log_metrics(m_lr)

    # --- Optuna on LightGBM: class weighting is part of the search
    dtr = lgb.Dataset(Xtr, ytr, params={"feature_pre_filter": False})  # min_child_samples varies per trial
    dva = lgb.Dataset(Xva, yva, reference=dtr)

    def objective(trial):
        params = dict(
            objective="binary", metric="average_precision", verbosity=-1, seed=42,
            learning_rate=0.05,
            num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            feature_fraction=trial.suggest_float("feature_fraction", 0.3, 0.9),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=1,
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            scale_pos_weight=trial.suggest_categorical("scale_pos_weight", [1.0, 3.0, 10.0]),
        )
        b = lgb.train(params, dtr, 1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        trial.set_user_attr("best_iter", b.best_iteration)
        return average_precision_score(yva, b.predict(Xva, num_iteration=b.best_iteration))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_params
    n_iter = study.best_trial.user_attrs["best_iter"]
    print(f"best val PR-AUC={study.best_value:.4f} params={best} iters={n_iter}")

    params = dict(objective="binary", verbosity=-1, seed=42, learning_rate=0.05, bagging_freq=1, **best)
    booster = lgb.train(params, dtr, n_iter)
    thr, _ = best_threshold(yva, booster.predict(Xva), amt_va)
    with mlflow.start_run(run_name="lightgbm_tuned"):
        mlflow.log_params({**best, "n_iter": n_iter, "threshold": thr})
        m = evaluate("lightgbm@replay", yre, booster.predict(Xre), amt_re, thr)
        mlflow.log_metrics(m)
        base = m_lr["expected_cost"]
        saving = 1 - m["expected_cost"] / base
        mlflow.log_metric("cost_saving_vs_baseline", saving)
        print(f"expected-cost saving vs baseline: {saving:.1%}")
        booster.save_model(str(PROCESSED / "model_v1.txt"))  # write_text mangles newlines on Windows
        PROCESSED.joinpath("model_v1.json").write_text(
            json.dumps({"threshold": thr, "features": list(Xtr.columns), "params": best, "n_iter": n_iter})
        )
        mlflow.log_artifact(str(PROCESSED / "model_v1.json"))


def make_lr(num_cols):
    pre = ColumnTransformer(
        [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols)],
        remainder="drop",
    )
    return make_pipeline(pre, LogisticRegression(max_iter=1000, class_weight="balanced"))


if __name__ == "__main__":
    main()
