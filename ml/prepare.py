"""Stage 3 of the lab: load monthly CSVs, clean, split 70/15/15."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flatprice.features import INPUT_FIELDS  # noqa: E402

DATA = ROOT / "ml" / "data"
PATTERNS = {"sale": "apartments_pl_*.csv", "rent": "apartments_rent_pl_*.csv"}


def load(target: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA / PATTERNS[target])))
    if not files:
        raise SystemExit(f"No files for {target} in {DATA}. See README (Kaggle download).")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["month"] = Path(f).stem[-7:]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # The same listing reappears every month: keep the latest snapshot only,
    # otherwise copies of one flat land in both train and test (leakage).
    df = df.sort_values("month").drop_duplicates("id", keep="last")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.price > 0]
    lo, hi = df.price.quantile([0.005, 0.995])
    df = df[(df.price >= lo) & (df.price <= hi)]
    df = df[(df.squareMeters >= 10) & (df.squareMeters <= 300)]
    return df[INPUT_FIELDS + ["price"]].reset_index(drop=True)


def split(df: pd.DataFrame):
    train, tmp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(tmp, test_size=0.5, random_state=42)
    return train, val, test


if __name__ == "__main__":
    for t in PATTERNS:
        d = clean(load(t))
        tr, va, te = split(d)
        print(t, "rows:", len(d), "train/val/test:", len(tr), len(va), len(te))
