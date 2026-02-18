from pydantic import BaseModel, Field
from typing import List, Optional


# ── Requests ───────────────────────────────────────────────────────────────────

class PredictManualRequest(BaseModel):
    """Predição com histórico de preços fornecido manualmente."""
    prices: List[float] = Field(
        ...,
        min_length=60,
        max_length=60,
        description="Lista com exatamente 60 preços de fechamento (do mais antigo ao mais recente)",
    )
    symbol: Optional[str] = Field(
        default="PETR4.SA",
        description="Símbolo da ação (apenas para identificação na resposta)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "PETR4.SA",
                "prices": [28.5] * 30 + [30.0] * 30,
            }
        }
    }


class PredictLiveRequest(BaseModel):
    """Predição com dados buscados automaticamente do Yahoo Finance."""
    symbol: str = Field(
        default="PETR4.SA",
        description="Símbolo da ação no Yahoo Finance (ex: PETR4.SA, VALE3.SA)",
    )

    model_config = {
        "json_schema_extra": {"example": {"symbol": "PETR4.SA"}}
    }


class ForecastRequest(BaseModel):
    """Previsão de múltiplos dias à frente."""
    symbol: str = Field(
        default="PETR4.SA",
        description="Símbolo da ação no Yahoo Finance",
    )
    days: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Número de dias úteis a prever (1–30)",
    )

    model_config = {
        "json_schema_extra": {"example": {"symbol": "PETR4.SA", "days": 5}}
    }


# ── Responses ──────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Resposta de uma predição de próximo dia."""
    symbol: str
    predicted_price: float = Field(description="Preço previsto para o próximo dia útil (R$)")
    predicted_ratio: float = Field(description="Ratio normalizado previsto pelo modelo (~1.0)")
    reference_price: float = Field(description="Preço de referência (início da janela de 60 dias)")
    last_known_price: float = Field(description="Último preço de fechamento conhecido (R$)")
    expected_change_pct: float = Field(description="Variação esperada em % em relação ao último preço")
    prediction_for_date: str = Field(description="Data alvo da previsão (próximo dia útil)")
    last_data_date: str = Field(description="Data do último dado utilizado")
    inference_time_ms: float = Field(description="Tempo de inferência do modelo (ms)")
    timestamp: str = Field(description="Timestamp UTC da requisição")


class ForecastDay(BaseModel):
    """Previsão de um dia específico no forecast multi-step."""
    day: int
    date: str
    predicted_price: float
    expected_change_pct: float


class ForecastResponse(BaseModel):
    """Resposta de previsão multi-step."""
    symbol: str
    base_price: float = Field(description="Preço base (último dia com dados reais)")
    base_date: str = Field(description="Data do último dado real")
    forecast_days: int
    forecast: List[ForecastDay]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    symbol: str
    architecture: str
    look_back: int
    normalization: str
    training_period: str
    splits: dict
    metrics_test: dict
    optimizer: str
    loss_function: str
    epochs_trained: int
    tf_version: str
    created_at: str
    model_load_time_ms: float
