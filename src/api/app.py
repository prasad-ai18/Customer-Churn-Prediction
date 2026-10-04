import time
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from src.logger import logger
from src.exceptions import ChurnPredictionException
from src.api.routes import router as api_router

FRONTEND_DIR = settings.BASE_DIR / "frontend"

def create_app() -> FastAPI:
    """Application factory for FastAPI serving."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production Customer Churn Intelligence & Prediction Platform. "
            "Delivers real-time churn probability, risk level classification, "
            "and SHAP-powered local explainability."
        ),
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration
    origins = settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if "*" not in origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Structured request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        try:
            response = await call_next(request)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration_ms}ms) from {client_ip}")
            return response
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(f"{request.method} {request.url.path} failed ({duration_ms}ms): {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "InternalServerError",
                    "message": "An unexpected error occurred while processing the request.",
                    "path": request.url.path
                }
            )

    # Custom exception handler for domain exceptions
    @app.exception_handler(ChurnPredictionException)
    async def custom_exception_handler(request: Request, exc: ChurnPredictionException):
        logger.warning(f"Domain exception on {request.url.path}: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details
            }
        )

    # Include REST API routes with /api prefix as well as top-level endpoints matching SOP requirements
    app.include_router(api_router, prefix="/api")
    app.include_router(api_router)

    # Serve static frontend files if directory exists
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app

app = create_app()
