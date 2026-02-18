import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.config import API_DESCRIPTION, API_TITLE, API_VERSION, METADATA_PATH, MODEL_PATH
from app.middleware.metrics import MetricsMiddleware
from app.routers import health, monitoring, predict
from app.services.model_service import ModelService

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== API startup — loading LSTM model ===")
    app.state.model_service = ModelService(MODEL_PATH, METADATA_PATH)
    logger.info("=== API ready — model loaded successfully ===")
    yield
    logger.info("=== API shutdown ===")


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS — permite acesso de qualquer origem (ajuste para produção se necessário)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de métricas (Prometheus)
app.add_middleware(MetricsMiddleware)

# Endpoint /metrics no formato Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router, prefix="/predict", tags=["Predictions"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])
