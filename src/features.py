"""Offline rolling-velocity features per entity (card, device, email domain).

Windows only look at PAST events (closed="left"), so the value stored for a transaction is what
the streaming consumer can know before that transaction arrives. This is the contract the
Week 2 online consumer must reproduce (feature parity test).
"""
import pandas as pd

from src.data import PROCESSED, TIME_COL

# TransactionDT is seconds from an unknown reference; any fixed epoch works for ordering.
EPOCH = pd.Timestamp("2017-12-01", tz="UTC")
WINDOWS = {"1h": "1h", "24h": "24h"}

ENTITIES = {
    "card": "card_id",
    "device": "device_info",
    "email": "email_domain",
}


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_timestamp"] = EPOCH + pd.to_timedelta(df[TIME_COL], unit="s")
    df["card_id"] = (
        df[["card1", "card2", "card3", "card5"]].astype("string").fillna("na").agg("_".join, axis=1)
    )
    df["device_info"] = df["DeviceInfo"].astype("string")
    df["email_domain"] = df["P_emaildomain"].astype("string")
    return df


def velocity(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Past-only count and amount sum per key over each window. Rows with null key are dropped."""
    d = df.loc[df[key].notna(), ["TransactionID", key, "event_timestamp", "TransactionAmt"]]
    d = d.sort_values([key, "event_timestamp"]).set_index("event_timestamp")
    out = d[["TransactionID", key]].copy()
    grouped = d.groupby(key)["TransactionAmt"]
    for name, win in WINDOWS.items():
        roll = grouped.rolling(win, closed="left")
        out[f"{prefix}_txn_count_{name}"] = roll.count().fillna(0).to_numpy()
        out[f"{prefix}_amt_sum_{name}"] = roll.sum().fillna(0.0).to_numpy()
    return out.reset_index().sort_values("event_timestamp")


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Join the precomputed offline velocity features onto a split by TransactionID."""
    for prefix, key in ENTITIES.items():
        f = pd.read_parquet(PROCESSED / f"features_{prefix}.parquet").drop(columns=[key, "event_timestamp"])
        df = df.merge(f, on="TransactionID", how="left")
    return df


def build() -> None:
    parts = [pd.read_parquet(PROCESSED / f"{n}.parquet") for n in ("train", "val", "replay")]
    df = add_keys(pd.concat(parts, ignore_index=True))
    for prefix, key in ENTITIES.items():
        feats = velocity(df, key, prefix)
        feats.to_parquet(PROCESSED / f"features_{prefix}.parquet", index=False)
        print(f"{prefix}: {len(feats):,} rows, {feats[key].nunique():,} entities")


if __name__ == "__main__":
    build()
