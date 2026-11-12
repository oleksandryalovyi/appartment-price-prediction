from __future__ import annotations

import math

from flatprice.features import DISTANCES, INPUT_FIELDS, encode_records

from .models import ModelArtifacts
from .predictors import Predictor
from .schemas import PredictRequest, PredictResponse, Prediction


class InputError(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__("invalid input")
        self.errors = errors


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


class PriceService:
    def __init__(self, models: dict[str, ModelArtifacts], predictor: Predictor, predictor_name: str):
        self.models = models
        self.predictor = predictor
        self.predictor_name = predictor_name
        # Rent and sale share cities; sale has the broader set of categorical values.
        self.meta = models.get("sale", next(iter(models.values()))).meta

    def public_meta(self) -> dict:
        cats: dict[str, set] = {}
        for m in self.models.values():
            for k, v in m.categories.items():
                cats.setdefault(k, set()).update(v)
        return {
            "categories": {k: sorted(v) for k, v in cats.items()},
            "cities": self.meta["cities"],
            "targets": {t: {"mae": m.metrics["model"]["mae"], "r2": m.metrics["model"]["r2"],
                            "trainedAt": m.metrics.get("trained_at")} for t, m in self.models.items()},
        }

    def _validate(self, req: PredictRequest) -> None:
        cats = self.public_meta()["categories"]
        errors = []
        for field in ("city", "type", "ownership", "buildingMaterial", "condition"):
            v = getattr(req, field)
            if v is not None and v not in cats[field]:
                errors.append({"loc": ["body", field], "msg": f"must be one of {cats[field]}", "type": "value_error"})
        for t in req.targets:
            if t not in self.models:
                errors.append({"loc": ["body", "targets"], "msg": f"model '{t}' is not available", "type": "value_error"})
        if errors:
            raise InputError(errors)

    def _impute(self, record: dict) -> dict[str, float]:
        imputed = {}
        city = self.meta["cities"].get(record["city"])
        if not city:
            return imputed
        if record.get("centreDistance") is None and record.get("latitude") is not None and record.get("longitude") is not None:
            imputed["centreDistance"] = round(haversine_km((record["latitude"], record["longitude"]), tuple(city["centre"])), 2)
        for c in DISTANCES:
            if record.get(c) is None and c not in imputed:
                imputed[c] = city["medians"][c]
        record.update(imputed)
        return imputed

    def predict(self, req: PredictRequest) -> PredictResponse:
        self._validate(req)
        record = {k: getattr(req, k) for k in INPUT_FIELDS}
        imputed = self._impute(record) if req.fillMissing else {}
        out: dict = {"imputed": imputed, "predictor": self.predictor_name}
        for t in req.targets:
            m = self.models[t]
            row = encode_records([record], m.categories, m.columns)
            price = max(float(self.predictor.predict(t, row)[0]), 0.0)
            # Per-city test MAE is a more honest range than the global one (small cities differ).
            mae = float(m.metrics.get("mae_by_city", {}).get(req.city, m.metrics["model"]["mae"]))
            out[t] = Prediction(price=round(price), pricePerM2=round(price / req.squareMeters, 1),
                                low=round(max(price - mae, 0)), high=round(price + mae), mae=round(mae))
        return PredictResponse(**out)
