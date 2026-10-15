"""Load, merge, validate and time-split the IEEE-CIS Fraud Detection data."""
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
TARGET = "isFraud"
TIME_COL = "TransactionDT"

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remaining 15% is the "live" replay stream

raw_schema = pa.DataFrameSchema(
    {
        "TransactionID": pa.Column(int, unique=True),
        TARGET: pa.Column(int, pa.Check.isin([0, 1])),
        TIME_COL: pa.Column(int, pa.Check.ge(0)),
        "TransactionAmt": pa.Column(float, pa.Check.gt(0)),
    },
    strict=False,
    coerce=True,
)


def load_merged() -> pd.DataFrame:
    tx = pd.read_csv(RAW / "train_transaction.csv")
    ident = pd.read_csv(RAW / "train_identity.csv")
    df = tx.merge(ident, on="TransactionID", how="left")
    raw_schema.validate(df)
    return df.sort_values(TIME_COL).reset_index(drop=True)


def time_split(df: pd.DataFrame):
    """Oldest 70% train, next 15% validation, newest 15% replay. Never random."""
    n = len(df)
    a, b = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def profile(df: pd.DataFrame) -> None:
    print(f"rows={len(df):,} cols={df.shape[1]}")
    print(f"fraud rate={df[TARGET].mean():.4%}")
    miss = df.isna().mean().sort_values(ascending=False)
    print(f"columns >50% missing: {(miss > 0.5).sum()} of {df.shape[1]}")


def prepare() -> None:
    df = load_merged()
    profile(df)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in zip(("train", "val", "replay"), time_split(df)):
        part.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):,} rows, fraud={part[TARGET].mean():.4%}")


if __name__ == "__main__":
    prepare()
