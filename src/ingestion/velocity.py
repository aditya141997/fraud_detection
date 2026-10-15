"""Streaming rolling-velocity state. Must match src/features.py exactly (the parity contract):
for a transaction at time t, count/sum over earlier events with t - window <= ts < t."""
from collections import defaultdict, deque

WINDOWS = {"1h": 3600, "24h": 86400}
MAX_WINDOW = max(WINDOWS.values())


def _s(v):
    return None if v is None or v != v else str(v)  # NaN != NaN


def entity_keys(rec: dict) -> dict:
    """Entity key per view; None means the transaction has no such entity (skipped)."""
    card = "_".join(_s(rec.get(c)) or "na" for c in ("card1", "card2", "card3", "card5"))
    return {
        "card": card,
        "device": _s(rec.get("DeviceInfo")),
        "email": _s(rec.get("P_emaildomain")),
    }


class VelocityState:
    def __init__(self):
        self._events = {p: defaultdict(deque) for p in ("card", "device", "email")}

    def update(self, rec: dict) -> dict:
        """Features for this transaction (past events only), then record it. Time order assumed."""
        t, amt = rec["TransactionDT"], rec["TransactionAmt"]
        out = {}
        for prefix, key in entity_keys(rec).items():
            if key is None:
                continue
            q = self._events[prefix][key]
            while q and q[0][0] < t - MAX_WINDOW:
                q.popleft()
            feats = {}
            for name, secs in WINDOWS.items():
                cnt, total = 0, 0.0
                for ts, a in reversed(q):
                    if ts < t - secs:
                        break
                    if ts < t:
                        cnt += 1
                        total += a
                feats[f"{prefix}_txn_count_{name}"] = cnt
                feats[f"{prefix}_amt_sum_{name}"] = total
            out[prefix] = (key, feats)
            q.append((t, amt))
        return out
