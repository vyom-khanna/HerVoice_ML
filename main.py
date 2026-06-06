"""
HerVoice - FastAPI Application Entry Point
app/main.py

Lifespan:
  1. Ensure all DB tables exist (create_all).
  2. Load pre-trained ML model weights into application memory so that
     /api/heatmap/predict has zero cold-start latency on first request.
"""

import contextlib
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.database import engine, Base
from app.routes import router
from app.services.ml_service import safety_predictor   # module-level singleton

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Step 1: Ensure database schema is up to date ──────────────────────────
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified / created ✓")

    # ── Step 2: Load ML model weights into memory ─────────────────────────────
    # This happens once at startup. All subsequent calls to
    # safety_predictor.predict() are pure in-memory — no disk I/O.
    loaded = safety_predictor.load()
    if loaded:
        logger.info("SafetyPredictor model loaded into application memory ✓")
    else:
        logger.warning(
            "SafetyPredictor: no pre-trained weights found. "
            "The /api/heatmap/predict endpoint will return fallback scores until "
            "POST /api/ml/train is called or generate_data.py + train workflow is run."
        )

    yield  # ← application runs here

    # ── Shutdown (nothing to teardown for joblib models) ──────────────────────
    logger.info("HerVoice API shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="HerVoice API",
    version="2.0.0",
    description=(
        "Real-time crowd-sourced women's safety platform with "
        "Spatio-Temporal ML prediction layer."
    ),
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router)


@app.get("/")
def read_root():
    return {
        "message":     "HerVoice API is online",
        "version":     "2.0.0",
        "ml_ready":    safety_predictor.is_ready,
    }


@app.get("/health")
def health_check():
    return {
        "status":   "ok",
        "ml_ready": safety_predictor.is_ready,
    }
