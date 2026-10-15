"""Online (streaming) velocity features must equal the offline (Feast/Parquet) ones."""
import pandas as pd
import pytest

from src.data import PROCESSED
from src.ingestion.velocity import VelocityState

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "replay.parquet").exists(), reason="run `make data features` first"
)

N = 20_000


def test_online_equals_offline():
    df = pd.read_parquet(PROCESSED / "train.parquet").tail(N)  # sorted by TransactionDT
    state = VelocityState()
    online = {}
    for rec in df.to_dict("records"):
        for prefix, (_, feats) in state.update(rec).items():
            online[(prefix, rec["TransactionID"])] = feats

    checked = 0
    for prefix in ("card", "device", "email"):
        off = pd.read_parquet(PROCESSED / f"features_{prefix}.parquet").set_index("TransactionID")
        cols = [c for c in off.columns if "_txn_count_" in c or "_amt_sum_" in c]
        # state starts at the slice start, so only compare once its 24h history is complete
        warm = df[df.TransactionDT >= df.TransactionDT.min() + 86400].TransactionID
        for tid in warm[warm.isin(off.index)]:
            got = online[(prefix, tid)]
            for c in cols:
                assert got[c] == pytest.approx(off.loc[tid, c]), (prefix, tid, c)
            checked += 1
    assert checked > 1000
