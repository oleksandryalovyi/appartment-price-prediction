"""Feature schema + encoding shared by training and serving (no train/serve skew).

Encoding mirrors the lab: yes/no -> 1/0, one-hot for categories with an extra
`<col>_nan` column (like ``pd.get_dummies(dummy_na=True)``), numeric missing
values stay NaN (XGBoost handles them natively).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

NUMERIC = [
    "squareMeters", "rooms", "floor", "floorCount", "buildYear",
    "latitude", "longitude", "centreDistance", "poiCount",
    "schoolDistance", "clinicDistance", "postOfficeDistance",
    "kindergartenDistance", "restaurantDistance", "collegeDistance",
    "pharmacyDistance",
]
DISTANCES = [
    "centreDistance", "poiCount", "schoolDistance", "clinicDistance",
    "postOfficeDistance", "kindergartenDistance", "restaurantDistance",
    "collegeDistance", "pharmacyDistance",
]
YES_NO = ["hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]
CATEGORICAL = ["city", "type", "ownership", "buildingMaterial", "condition"]
INPUT_FIELDS = ["city", "type", *NUMERIC, "ownership", "buildingMaterial", "condition", *YES_NO]


def _is_missing(v: Any) -> bool:
    return v is None or v is pd.NA or v == "" or (isinstance(v, float) and math.isnan(v))


def _yes_no(v: Any) -> float:
    if _is_missing(v):
        return np.nan
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v != 0)
    return 1.0 if str(v).strip().lower() in ("yes", "true", "1") else 0.0


def build_columns(categories: Mapping[str, list[str]]) -> list[str]:
    cols = [*NUMERIC, *YES_NO]
    for c in CATEGORICAL:
        cols += [f"{c}_{v}" for v in categories[c]] + [f"{c}_nan"]
    return cols


def learn_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL}


def encode_frame(df: pd.DataFrame, categories: Mapping[str, list[str]], columns: list[str]) -> pd.DataFrame:
    """Encode raw records (dataset columns) into the model matrix, ordered by ``columns``."""
    out = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        out[c] = pd.to_numeric(df.get(c), errors="coerce").astype(float)
    for c in YES_NO:
        out[c] = df[c].map(_yes_no).astype(float) if c in df else np.nan
    for c in CATEGORICAL:
        vals = df[c].astype(object) if c in df else pd.Series([None] * len(df), index=df.index, dtype=object)
        missing = vals.map(_is_missing).astype(bool)
        vals = vals.where(~missing, "")
        known = set(categories[c])
        for v in categories[c]:
            out[f"{c}_{v}"] = (vals.astype(str) == v).astype(float)
        # Unknown or missing value -> the "nan" bucket, like dummy_na=True.
        out[f"{c}_nan"] = (missing | ~vals.astype(str).isin(known)).astype(float)
    return out[columns]


def encode_records(records: Iterable[Mapping[str, Any]], categories, columns) -> np.ndarray:
    df = pd.DataFrame(list(records))
    return encode_frame(df, categories, columns).to_numpy(dtype=float)


def to_csv_row(row: np.ndarray) -> str:
    """CSV row for SageMaker built-in XGBoost (empty string = missing)."""
    return ",".join("" if math.isnan(x) else repr(float(x)) for x in row)
