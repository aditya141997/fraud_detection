"""Promotion rule: when may the challenger replace the champion?

Inputs are the shadow-window comparison produced by `shadow_eval.compare` (both models scored the same live
events; labels joined afterwards). The challenger is promoted only if ALL of these hold:

1. enough evidence: at least `min_events` events and `min_frauds` frauds in the window;
2. it is cheaper: expected cost is at least `min_saving` (relative) below the champion's, AND the lower end
   of the paired-bootstrap `confidence` interval of that saving is above zero (a gain inside the noise
   does not count);
3. it does not rank worse: PR-AUC is at most `pr_auc_tolerance` below the champion's;
4. it does not behave oddly: its flag rate is within [1/max_flag_ratio, max_flag_ratio] of the champion's,
   so a threshold that suddenly flags (or stops flagging) most traffic cannot slip through on cost alone.

Anything else keeps the champion. The result lists every failed check so the decision can be audited.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionRule:
    min_events: int = 5_000
    min_frauds: int = 100
    min_saving: float = 0.02
    pr_auc_tolerance: float = 0.005
    max_flag_ratio: float = 2.0


def decide(cmp: dict, rule: PromotionRule = PromotionRule()) -> dict:
    """`cmp` is `shadow_eval.compare` output. Returns {"promote": bool, "reasons": [...failed checks]}."""
    champ, chal = cmp["champion"], cmp["challenger"]
    reasons = []
    if cmp["n_events"] < rule.min_events:
        reasons.append(f"only {cmp['n_events']} events (need {rule.min_events})")
    if cmp["n_frauds"] < rule.min_frauds:
        reasons.append(f"only {cmp['n_frauds']} frauds (need {rule.min_frauds})")
    if cmp["cost_saving"] < rule.min_saving:
        reasons.append(f"cost saving {cmp['cost_saving']:.1%} below {rule.min_saving:.0%}")
    if cmp["cost_saving_ci"][0] <= 0:
        reasons.append(f"saving not significant: CI lower bound {cmp['cost_saving_ci'][0]:.1%} <= 0")
    if chal["pr_auc"] < champ["pr_auc"] - rule.pr_auc_tolerance:
        reasons.append(f"PR-AUC {chal['pr_auc']:.3f} worse than champion {champ['pr_auc']:.3f}")
    ratio = chal["flag_rate"] / champ["flag_rate"] if champ["flag_rate"] else float("inf")
    if not (1 / rule.max_flag_ratio <= ratio <= rule.max_flag_ratio):
        reasons.append(f"flag rate ratio {ratio:.2f} outside [{1 / rule.max_flag_ratio:.2f}, {rule.max_flag_ratio:.1f}]")
    return {"promote": not reasons, "reasons": reasons}
