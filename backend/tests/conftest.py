import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import build_service, create_app
from app.service import PriceService


@pytest.fixture(scope="session")
def settings():
    return Settings()


@pytest.fixture(scope="session")
def real_service(settings) -> PriceService:
    return build_service(settings)


@pytest.fixture()
def client(real_service):
    with TestClient(create_app(service=real_service)) as c:
        yield c


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, target, rows):
        self.calls.append((target, rows))
        return [1000.0 if target == "rent" else 500000.0] * len(rows)


@pytest.fixture()
def fake_client(real_service):
    fake = FakePredictor()
    svc = PriceService(real_service.models, fake, "fake")
    with TestClient(create_app(service=svc)) as c:
        c.fake = fake
        yield c
