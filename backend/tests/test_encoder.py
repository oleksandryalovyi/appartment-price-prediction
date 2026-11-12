"""Train/serve parity: encoding + prediction in the API equals what train.py produced."""
import json
import math

import numpy as np
import pytest

from flatprice.features import encode_records, to_csv_row


@pytest.mark.parametrize("target", ["rent", "sale"])
def test_encoding_matches_training(real_service, target):
    m = real_service.models[target]
    fx = json.loads((m.dir / "fixture.json").read_text())
    got = encode_records(fx["records"], m.categories, m.columns)
    want = np.array(fx["encoded"], dtype=float)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(np.nan_to_num(got), np.nan_to_num(want))


@pytest.mark.parametrize("target", ["rent", "sale"])
def test_local_predictions_match_training(real_service, target):
    m = real_service.models[target]
    fx = json.loads((m.dir / "fixture.json").read_text())
    rows = encode_records(fx["records"], m.categories, m.columns)
    np.testing.assert_allclose(real_service.predictor.predict(target, rows), fx["predictions"], rtol=1e-5)


def test_one_hot_and_missing(real_service):
    m = real_service.models["rent"]
    row = encode_records([{"city": "krakow", "squareMeters": 50, "rooms": 2, "hasBalcony": True}], m.categories, m.columns)[0]
    col = dict(zip(m.columns, row))
    assert col["city_krakow"] == 1 and col["city_warszawa"] == 0 and col["city_nan"] == 0
    assert col["type_nan"] == 1  # missing category -> nan bucket (dummy_na=True)
    assert col["hasBalcony"] == 1
    assert math.isnan(col["floor"])  # numeric missing stays NaN for XGBoost


def test_csv_row_uses_empty_for_missing():
    assert to_csv_row(np.array([1.0, np.nan, 2.5])) == "1.0,,2.5"
