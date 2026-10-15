"""SHAP computation in a separate process.

TreeSHAP on the 1,490-tree model costs ~0.4 s of CPU per row. Run inside the scoring workers it made
concurrent /explain calls stall (a few requests hung for 20 s+ in tests with 6-16 clients), so it runs in
its own small process pool and the scoring workers only await the result. Importable without the app
(only lightgbm/numpy) because each pool process imports this module on start.
"""
import lightgbm as lgb
import numpy as np

from src.training.explain import top_reasons_array

_booster = None
_names = None


def init(model_file: str) -> None:
    global _booster, _names
    _booster = lgb.Booster(model_file=model_file)
    _names = _booster.feature_name()


def shap(row: np.ndarray, k: int) -> list[dict]:
    return top_reasons_array(_booster, row, _names, k)


def ping() -> bool:
    return _booster is not None
