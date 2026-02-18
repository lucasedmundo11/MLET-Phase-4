from datetime import datetime

from fastapi import APIRouter, Request

from app.schemas.prediction import HealthResponse

router = APIRouter()


@router.get(
    "/",
    summary="Informações da API",
    tags=["Health"],
)
async def root():
    """Endpoint raiz com informações básicas da API e links de navegação."""
    return {
        "service": "PETR4.SA LSTM Stock Price Predictor",
        "version": "1.0.0",
        "status": "running",
        "model": "LSTM — PETR4.SA (Petrobras)",
        "links": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/monitoring/model/info",
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificação de saúde da API",
    tags=["Health"],
)
async def health_check(request: Request):
    """
    Retorna o status de saúde da API.

    - **healthy**: modelo carregado e API pronta.
    - **degraded**: API iniciada mas modelo ainda não disponível.
    """
    model_loaded = (
        hasattr(request.app.state, "model_service")
        and request.app.state.model_service.model is not None
    )
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
