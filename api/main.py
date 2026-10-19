"""Run from the repository root using the privacy-preserving `make api` target."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from api.middleware import RequestGuard
from phishlens.config import ARTIFACT_DIR
from phishlens.features.url_online import UnsafeTarget, safe_domain
from phishlens.predict import Prediction, Predictor
from phishlens.schemas import EmailRequest, HealthResponse, PredictionResponse, URLRequest

logger = logging.getLogger("phishlens.api")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def create_app(
    artifact_dir: Path = ARTIFACT_DIR,
    predictor_factory: Callable[[Path], Predictor] = Predictor.from_directory,
    rate_limit: int = 30,
    rate_window: float = 60,
    clock: Callable[[], float] | None = None,
) -> FastAPI:
    """Load trusted local artifacts once per lifespan; serialize CPU inference per process."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = None
        app.state.inference_lock = asyncio.Lock()
        try:
            app.state.predictor = await run_in_threadpool(predictor_factory, artifact_dir)
        except Exception as exc:
            logger.error(
                json.dumps({"event": "model_load_failed", "error_type": type(exc).__name__})
            )
        yield
        app.state.predictor = None

    app = FastAPI(title="PhishLens", version="0.1.0", lifespan=lifespan)
    options = {"limit": rate_limit, "window": rate_window}
    if clock is not None:
        options["clock"] = clock
    app.add_middleware(RequestGuard, **options)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's default errors include input values, including entire email bodies.
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Invalid request: check required fields, types, content and size limits",
                "errors": [
                    {
                        "type": error["type"],
                        "field": next(
                            (part for part in error["loc"] if part in {"url", "mode", "text"}),
                            "body",
                        ),
                    }
                    for error in exc.errors()
                ],
            },
        )

    def predictor() -> Predictor:
        result = getattr(app.state, "predictor", None)
        if result is None:
            raise HTTPException(503, "Models unavailable; check trusted local artifacts")
        return result

    @app.get("/health", response_model=HealthResponse)
    async def health() -> JSONResponse:
        loaded = getattr(app.state, "predictor", None)
        models = {}
        for name, attribute in [
            ("url_offline", "offline"),
            ("url_online", "online"),
            ("email", "email"),
        ]:
            artifact = getattr(loaded, attribute, None)
            metadata = artifact["metadata"] if artifact else {}
            models[name] = {
                "version": metadata.get("version"),
                "available": artifact is not None,
                "experimental": bool(metadata.get("experimental", False)),
            }
        status = (
            "unavailable"
            if loaded is None
            else ("ok" if models["email"]["available"] else "degraded")
        )
        result = HealthResponse(status=status, models=models)
        return JSONResponse(result.model_dump(), status_code=503 if loaded is None else 200)

    async def predict(method: Callable[..., Prediction], *args: str) -> PredictionResponse:
        # Serial inference bounds memory and protects lazily initialized explainers.
        # The event loop remains free for health checks while CPU/WHOIS work runs.
        async with app.state.inference_lock:
            try:
                result = await run_in_threadpool(method, *args)
                return PredictionResponse.model_validate(asdict(result))
            except UnsafeTarget:
                raise HTTPException(422, "Online mode requires a safe public domain") from None
            except Exception as exc:
                logger.error(
                    json.dumps({"event": "prediction_failed", "error_type": type(exc).__name__})
                )
                raise HTTPException(500, "Prediction failed") from None

    @app.post("/predict/url", response_model=PredictionResponse)
    async def predict_url(body: URLRequest) -> PredictionResponse:
        if body.mode == "online":
            try:
                safe_domain(body.url)
            except ValueError:
                raise HTTPException(422, "Online mode requires a safe public domain") from None
        return await predict(predictor().predict_url, body.url, body.mode)

    @app.post("/predict/email", response_model=PredictionResponse)
    async def predict_email(body: EmailRequest) -> PredictionResponse:
        model = predictor()
        if model.email is None:
            raise HTTPException(503, "Email model unavailable")
        return await predict(model.predict_email, body.text)

    return app


app = create_app()
