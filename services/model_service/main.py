"""CLSA Model Service.

A lightweight FastAPI service that wraps the CLSA inference engine.
Receives generation requests from the API backend and returns
model outputs. Runs on its own process/container, typically with
GPU access.
"""

import logging
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from services.model_service.inference import CLSAEngine
from services.model_service.schemas import GenerateRequest, GenerateResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = CLSAEngine(
    device=os.getenv("CLSA_DEVICE", "cpu"),
    model_id=os.getenv("CLSA_MODEL_ID", "HuggingFaceTB/SmolLM2-135M"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    engine.load()
    yield


app = FastAPI(
    title="CLSA Model Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine.model is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        return engine.generate(request)
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))
