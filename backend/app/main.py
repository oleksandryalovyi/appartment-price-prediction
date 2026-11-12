from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .models import load_artifacts
from .schemas import PredictRequest, PredictResponse
from .service import InputError, PriceService


def build_service(settings: Settings) -> PriceService:
    models = load_artifacts(settings.artifacts_dir)
    if settings.predictor == "sagemaker":
        from .predictors import SageMakerPredictor

        predictor = SageMakerPredictor(settings.aws_region, {
            "rent": settings.sagemaker_rent_endpoint, "sale": settings.sagemaker_sale_endpoint})
    else:
        from .predictors import LocalPredictor

        predictor = LocalPredictor(models)
    return PriceService(models, predictor, settings.predictor)


def create_app(settings: Settings | None = None, service: PriceService | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or build_service(settings)
        yield

    app = FastAPI(title="Flat Price Analyzer", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(InputError)
    async def _input_error(_: Request, exc: InputError):
        return JSONResponse(status_code=422, content={"detail": exc.errors})

    @app.get("/api/health")
    def health(request: Request):
        return {"status": "ok", "predictor": request.app.state.service.predictor_name}

    @app.get("/api/meta")
    def meta(request: Request):
        return request.app.state.service.public_meta()

    @app.post("/api/predict", response_model=PredictResponse, response_model_exclude_none=True)
    def predict(req: PredictRequest, request: Request):
        return request.app.state.service.predict(req)

    static = settings.static_dir
    if static and Path(static).is_dir():
        app.mount("/assets", StaticFiles(directory=Path(static) / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            f = Path(static) / path
            return FileResponse(f if path and f.is_file() else Path(static) / "index.html")

    return app


app = create_app()
