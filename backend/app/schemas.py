from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Target = Literal["rent", "sale"]


class FlatInput(BaseModel):
    city: str
    type: str | None = None
    squareMeters: float = Field(ge=10, le=300)
    rooms: int = Field(ge=1, le=10)
    floor: int | None = Field(None, ge=0, le=60)
    floorCount: int | None = Field(None, ge=1, le=60)
    buildYear: int | None = Field(None, ge=1800, le=2030)
    latitude: float | None = Field(None, ge=49.0, le=55.0)
    longitude: float | None = Field(None, ge=14.0, le=24.5)
    centreDistance: float | None = Field(None, ge=0, le=50)
    poiCount: float | None = Field(None, ge=0, le=500)
    schoolDistance: float | None = Field(None, ge=0, le=20)
    clinicDistance: float | None = Field(None, ge=0, le=20)
    postOfficeDistance: float | None = Field(None, ge=0, le=20)
    kindergartenDistance: float | None = Field(None, ge=0, le=20)
    restaurantDistance: float | None = Field(None, ge=0, le=20)
    collegeDistance: float | None = Field(None, ge=0, le=20)
    pharmacyDistance: float | None = Field(None, ge=0, le=20)
    ownership: str | None = None
    buildingMaterial: str | None = None
    condition: str | None = None
    hasParkingSpace: bool | None = None
    hasBalcony: bool | None = None
    hasElevator: bool | None = None
    hasSecurity: bool | None = None
    hasStorageRoom: bool | None = None

    @model_validator(mode="after")
    def _floor_le_count(self):
        if self.floor is not None and self.floorCount is not None and self.floor > self.floorCount:
            raise ValueError("floor cannot be greater than floorCount")
        return self


class PredictRequest(FlatInput):
    targets: list[Target] = ["rent", "sale"]
    fillMissing: bool = Field(True, description="Impute centreDistance from coordinates and other distances from city medians")


class Prediction(BaseModel):
    price: float
    pricePerM2: float
    low: float
    high: float
    mae: float


class PredictResponse(BaseModel):
    rent: Prediction | None = None
    sale: Prediction | None = None
    imputed: dict[str, float]
    predictor: str
