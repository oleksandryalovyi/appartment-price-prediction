"""Inference drivers. Both receive rows already ordered by columns.json."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from flatprice.features import to_csv_row


class Predictor(Protocol):
    def predict(self, target: str, rows: np.ndarray) -> list[float]: ...


class LocalPredictor:
    """Runs the exported XGBoost model in-process (free to host)."""

    def __init__(self, models: dict):
        import xgboost as xgb

        self._xgb = xgb
        self._boosters = {}
        for target, m in models.items():
            b = xgb.Booster()
            b.load_model(m.dir / "model.json")
            self._boosters[target] = (b, m.columns)

    def predict(self, target, rows):
        booster, columns = self._boosters[target]
        d = self._xgb.DMatrix(rows, feature_names=columns)
        return booster.predict(d).astype(float).tolist()


class SageMakerPredictor:
    """Calls the lab's SageMaker (Serverless) endpoint with text/csv, like the Node.js example."""

    def __init__(self, region: str, endpoints: dict[str, str | None]):
        import boto3

        self._client = boto3.client("sagemaker-runtime", region_name=region)
        self._endpoints = endpoints

    def predict(self, target, rows):
        endpoint = self._endpoints.get(target)
        if not endpoint:
            raise RuntimeError(f"No SageMaker endpoint configured for '{target}'")
        body = "\n".join(to_csv_row(r) for r in rows)
        res = self._client.invoke_endpoint(EndpointName=endpoint, ContentType="text/csv", Body=body)
        text = res["Body"].read().decode().strip()
        return [float(x) for x in text.replace("\n", ",").split(",") if x]
