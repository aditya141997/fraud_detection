"""Compare online (Redis) card features against offline values for the last N replayed events."""
import sys

import pandas as pd
from feast import FeatureStore

from src.data import PROCESSED
from src.features import add_keys

COLS = ["card_txn_count_1h", "card_amt_sum_1h", "card_txn_count_24h", "card_amt_sum_24h"]


def main(n_events: int = 3000, sample: int = 200):
    fs = FeatureStore("feature_repo")
    df = add_keys(pd.read_parquet(PROCESSED / "replay.parquet").head(n_events))
    last = df.groupby("card_id").tail(1)
    last = last.sample(min(sample, len(last)), random_state=1)
    off = pd.read_parquet(PROCESSED / "features_card.parquet").set_index("TransactionID")
    on = fs.get_online_features(
        features=[f"card_velocity:{c}" for c in COLS],
        entity_rows=[{"card_id": k} for k in last.card_id],
    ).to_df()
    exp = off.loc[last.TransactionID, COLS].reset_index(drop=True).astype(float)
    got = on[COLS].reset_index(drop=True).astype(float)
    print("rows:", len(got), "nulls:", int(got.isna().sum().sum()))
    print("max abs diff:", float((got - exp).abs().max().max()))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
