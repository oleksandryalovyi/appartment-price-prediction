"""Stages 4-5 of the lab, locally: train XGBoost, evaluate vs median baseline, export artifacts.

Usage: python ml/train.py --target rent|sale [--write-sagemaker-csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))
from flatprice.features import (  # noqa: E402
    DISTANCES, NUMERIC, build_columns, encode_frame, learn_categories,
)
from prepare import clean, load, split  # noqa: E402

PARAMS = dict(objective="reg:squarederror", eval_metric="rmse", max_depth=7, eta=0.1,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=3, seed=42)


def metrics(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(mean_squared_error(y, p) ** 0.5),
            "r2": float(r2_score(y, p))}


def build_meta(df: pd.DataFrame, categories) -> dict:
    cities = {}
    for city, g in df.groupby("city"):
        near = g[g.centreDistance <= g.centreDistance.quantile(0.02)]
        cities[city] = {
            "centre": [float(near.latitude.median()), float(near.longitude.median())],
            "medians": {c: float(g[c].median()) for c in DISTANCES},
            "count": int(len(g)),
        }
    ranges = {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC}
    return {"categories": categories, "cities": cities, "ranges": ranges}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["rent", "sale"], required=True)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--write-sagemaker-csv", action="store_true",
                    help="also write train/validation/test CSVs (target first, no header) for SageMaker")
    args = ap.parse_args()

    df = clean(load(args.target))
    train, val, test = split(df)
    categories = learn_categories(train)
    columns = build_columns(categories)
    enc = lambda d: encode_frame(d, categories, columns)  # noqa: E731
    Xtr, Xva, Xte = enc(train), enc(val), enc(test)

    dtrain = xgb.DMatrix(Xtr, label=train.price, feature_names=columns)
    dval = xgb.DMatrix(Xva, label=val.price, feature_names=columns)
    booster = xgb.train(PARAMS, dtrain, num_boost_round=args.rounds,
                        evals=[(dtrain, "train"), (dval, "validation")],
                        early_stopping_rounds=50, verbose_eval=250)

    preds = booster.predict(xgb.DMatrix(Xte, feature_names=columns),
                            iteration_range=(0, booster.best_iteration + 1))
    baseline = np.full(len(test), train.price.median())
    m = {"model": metrics(test.price, preds), "baseline_median": metrics(test.price, baseline),
         "best_iteration": int(booster.best_iteration), "rows": {"train": len(train), "validation": len(val), "test": len(test)},
         "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    # Per-city MAE (shows the "small city" effect from the lab's bonus experiment).
    m["mae_by_city"] = {c: float(mean_absolute_error(g.price, preds[g.index.map(test.index.get_loc)]))
                        for c, g in test.groupby("city")}
    print(json.dumps({k: m[k] for k in ("model", "baseline_median", "best_iteration")}, indent=2))

    out = ROOT / "artifacts" / args.target
    out.mkdir(parents=True, exist_ok=True)
    # Trim to best iteration so serving needs no iteration_range.
    booster = booster[: booster.best_iteration + 1]
    booster.save_model(out / "model.ubj")  # binary: smaller file, ~2x less RAM to load than JSON
    (out / "columns.json").write_text(json.dumps(columns, indent=1))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    (out / "meta.json").write_text(json.dumps(build_meta(df, categories), indent=1))
    # Fixture for backend parity tests: raw records + expected predictions.
    sample = test.head(5)
    fixture = {"records": json.loads(sample.drop(columns="price").to_json(orient="records")),
               "encoded": json.loads(enc(sample).to_json(orient="values")),
               "predictions": booster.predict(xgb.DMatrix(enc(sample), feature_names=columns)).tolist()}
    (out / "fixture.json").write_text(json.dumps(fixture, indent=1))

    if args.write_sagemaker_csv:
        sm = ROOT / "ml" / "data" / "sagemaker" / args.target
        sm.mkdir(parents=True, exist_ok=True)
        for name, d, X in (("train", train, Xtr), ("validation", val, Xva), ("test", test, Xte)):
            pd.concat([d.price.reset_index(drop=True), X.reset_index(drop=True)], axis=1) \
              .to_csv(sm / f"{name}.csv", header=False, index=False)
        print("SageMaker CSVs in", sm)
    print("Artifacts written to", out)


if __name__ == "__main__":
    main()
