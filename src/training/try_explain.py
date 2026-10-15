"""Smoke test: top-3 SHAP reasons for the highest-scoring replay transactions."""
import json

import lightgbm as lgb
import pandas as pd

from src.data import PROCESSED
from src.training.explain import top_reasons
from src.training.train import split_xy
from src.training.tune import load

meta = json.loads((PROCESSED / "model_v1.json").read_text())
booster = lgb.Booster(model_file=str(PROCESSED / "model_v1.txt"))
train = load("train")
Xtr, _ = split_xy(train)
cats = list(Xtr.select_dtypes("category").columns)
X, y = split_xy(load("replay").head(5000))
for c in cats:
    X[c] = pd.Categorical(X[c], categories=Xtr[c].cat.categories)
X = X[meta["features"]]
score = booster.predict(X)
top = pd.Series(score).nlargest(3).index
for i, reasons in zip(top, top_reasons(booster, X.iloc[top])):
    print(f"score={score[i]:.3f} actual={y.iloc[i]} threshold={meta['threshold']:.3f}")
    for r in reasons:
        print("   ", r)
