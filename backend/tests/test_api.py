BASE = {"city": "czestochowa", "type": "blockOfFlats", "squareMeters": 52, "rooms": 2,
        "floor": 3, "floorCount": 5, "buildYear": 1995, "latitude": 50.81, "longitude": 19.12,
        "ownership": "condominium", "buildingMaterial": "brick", "hasBalcony": True, "hasElevator": False}


def test_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert "czestochowa" in meta["categories"]["city"]
    assert set(meta["targets"]) == {"rent", "sale"}


def test_predict_real_models(client):
    r = client.post("/api/predict", json=BASE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 800 < body["rent"]["price"] < 6000
    assert 150_000 < body["sale"]["price"] < 1_500_000
    assert body["rent"]["low"] <= body["rent"]["price"] <= body["rent"]["high"]
    assert "centreDistance" in body["imputed"] and "schoolDistance" in body["imputed"]


def test_bigger_flat_costs_more(client):
    small = client.post("/api/predict", json=BASE | {"squareMeters": 35, "rooms": 1}).json()
    big = client.post("/api/predict", json=BASE | {"squareMeters": 90, "rooms": 4}).json()
    assert big["sale"]["price"] > small["sale"]["price"]


def test_single_target_and_no_imputation(fake_client):
    r = fake_client.post("/api/predict", json=BASE | {"targets": ["rent"], "fillMissing": False})
    body = r.json()
    assert "sale" not in body and body["rent"]["price"] == 1000 and body["imputed"] == {}
    assert [t for t, _ in fake_client.fake.calls] == ["rent"]


def test_validation_errors(fake_client):
    assert fake_client.post("/api/predict", json=BASE | {"floor": 7, "floorCount": 5}).status_code == 422
    assert fake_client.post("/api/predict", json=BASE | {"city": "berlin"}).status_code == 422
    assert fake_client.post("/api/predict", json=BASE | {"type": "castle"}).status_code == 422
    assert fake_client.post("/api/predict", json={k: v for k, v in BASE.items() if k != "squareMeters"}).status_code == 422
