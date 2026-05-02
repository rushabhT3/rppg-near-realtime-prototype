import asyncio
import threading

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import setup_routes

setup_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    print(f"CORS Allowed Origins: {settings.CORS_ALLOWED_ORIGINS}")
    preload_model()
    yield


# Create FastAPI application
app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_routes(app)


def preload_model():
    """Pre-load the JAX model in a background thread so the first request isn't blocked."""
    import logging

    logger = logging.getLogger("VITALIS_BACKEND")

    def _load():
        logger.info("Pre-loading rPPG model in background thread...")
        try:
            from app.services.analyzer import get_analyzer

            analyzer = get_analyzer()
            if analyzer.ready:
                logger.info("Model pre-loaded successfully!")
            else:
                logger.warning("Model pre-load failed - will retry on first request")
        except Exception as e:
            logger.error(f"Model pre-load error: {e}")

    t = threading.Thread(target=_load, daemon=True)
    t.start()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch ALL exceptions so errors always return JSON with CORS headers."""
    import traceback

    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )
