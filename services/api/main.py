"""CLSA API Backend.

Public-facing API that the frontend communicates with. Handles
conversation management, caching, and proxies generation requests
to the model service.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.cache import close_redis
from services.api.database import init_db
from services.api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB and Redis on startup, clean up on shutdown."""
    logger.info("Initializing database")
    await init_db()
    yield
    logger.info("Closing Redis connection")
    await close_redis()


app = FastAPI(
    title="CLSA API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
