"""Business cost of decisions. A missed fraud loses the transaction amount; a false alarm costs a
fixed manual-review fee. The threshold is chosen to minimise this cost, never fixed at 0.5."""
import numpy as np

REVIEW_COST = 5.0  # cost of reviewing/declining one legitimate transaction


def expected_cost(y, score, amount, threshold) -> float:
    flagged = score >= threshold
    missed = (~flagged) & (y == 1)
    false_alarm = flagged & (y == 0)
    return float(amount[missed].sum() + REVIEW_COST * false_alarm.sum())


def best_threshold(y, score, amount, grid=None):
    y, score, amount = map(np.asarray, (y, score, amount))
    grid = np.quantile(score, np.linspace(0.80, 0.9995, 400)) if grid is None else grid
    costs = np.array([expected_cost(y, score, amount, t) for t in grid])
    i = int(costs.argmin())
    return float(grid[i]), float(costs[i])


def no_model_cost(y, amount) -> float:
    """Cost of flagging nothing: every fraud is missed."""
    y, amount = np.asarray(y), np.asarray(amount)
    return float(amount[y == 1].sum())
