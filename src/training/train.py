"""Week 1: logistic regression baseline vs first LightGBM, compared on PR-AUC. Logged to MLflow."""
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import PROCESSED, TARGET, TIME_COL

DROP = [TARGET, "TransactionID", TIME_COL]


def load():
    train = pd.read_parquet(PROCESSED / "train.parquet")
    val = pd.read_parquet(PROCESSED / "val.parquet")
    return train, val


def split_xy(df):
    X = df.drop(columns=DROP)
    for c in X.select_dtypes("object"):
        X[c] = X[c].astype("category")
    return X, df[TARGET]


def recall_at_fpr(y, score, fpr_target=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.interp(fpr_target, fpr, tpr))


def report(name, y, score) -> dict:
    m = {"pr_auc": average_precision_score(y, score), "recall_at_1pct_fpr": recall_at_fpr(y, score)}
    print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
    return m


def main():
    mlflow.set_experiment("fraud-week1")
    train, val = load()
    Xtr, ytr = split_xy(train)
    Xva, yva = split_xy(val)
    cat_cols = list(Xtr.select_dtypes("category").columns)
    for c in cat_cols:  # align categories across splits
        Xva[c] = pd.Categorical(Xva[c], categories=Xtr[c].cat.categories)
    num_cols = [c for c in Xtr.columns if c not in cat_cols]

    with mlflow.start_run(run_name="logreg_baseline"):
        pre = ColumnTransformer(
            [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols)],
            remainder="drop",
        )
        lr = make_pipeline(pre, LogisticRegression(max_iter=1000, class_weight="balanced"))
        lr.fit(Xtr, ytr)
        mlflow.log_metrics(report("logreg", yva, lr.predict_proba(Xva)[:, 1]))

    with mlflow.start_run(run_name="lightgbm_v1"):
        params = dict(n_estimators=500, learning_rate=0.05, num_leaves=64,
                      subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        gbm = lgb.LGBMClassifier(**params)
        gbm.fit(Xtr, ytr)
        mlflow.log_params(params)
        mlflow.log_metrics(report("lightgbm", yva, gbm.predict_proba(Xva)[:, 1]))


if __name__ == "__main__":
    main()
