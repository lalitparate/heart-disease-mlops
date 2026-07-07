"""
FastAPI application — /predict endpoint with Prometheus monitoring.
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import (
    PatientFeatures, BatchRequest,
    PredictionResponse, BatchResponse, HealthResponse,
)
from src.predict import predict_single, predict_batch, load_model

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan: pre-load model at startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    try:
        load_model()
        app.state.model_loaded = True
        logger.info("Model loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(f"Model not found on startup: {exc}")
        app.state.model_loaded = False
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "MLOps Assignment 01 — AIMLCZG523\n\n"
        "Predicts the presence of heart disease from patient health features.\n"
        "Send a POST to `/predict` with 13 clinical features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)


# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 4)
    logger.info(
        f"{request.method} {request.url.path}  "
        f"status={response.status_code}  duration={duration}s"
    )
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Heart Disease Prediction API is running. Visit /docs for Swagger UI."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    loaded = getattr(app.state, "model_loaded", False)
    return HealthResponse(
        status="healthy" if loaded else "degraded — model not loaded",
        model_loaded=loaded,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: PatientFeatures):
    """
    Predict heart disease risk for a single patient.

    - **prediction**: 0 = No Disease, 1 = Disease Present
    - **probability**: confidence score (0–1)
    - **label**: human-readable result
    """
    try:
        result = predict_single(features.model_dump())
        logger.info(f"Prediction: {result}")
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch_endpoint(batch: BatchRequest):
    """
    Predict heart disease risk for multiple patients in one request.
    Maximum 500 records per call.
    """
    try:
        records = [r.model_dump() for r in batch.records]
        results = predict_batch(records)
        return BatchResponse(
            predictions=[PredictionResponse(**r) for r in results],
            count=len(results),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


# ── Error handlers ────────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Endpoint not found."})


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
