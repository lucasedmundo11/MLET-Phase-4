import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from app.middleware.metrics import PREDICTION_COUNT, PREDICTION_DURATION
from app.schemas.prediction import (
    ForecastRequest,
    ForecastResponse,
    ForecastDay,
    PredictLiveRequest,
    PredictManualRequest,
    PredictionResponse,
)
from app.services.data_service import fetch_prices, next_business_day

router = APIRouter()
logger = logging.getLogger(__name__)


# ── POST /predict ──────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predição — entrada manual",
    description=(
        "Prediz o preço de fechamento do próximo dia útil a partir de "
        "**60 preços históricos fornecidos pelo usuário** (ordenados do mais antigo "
        "ao mais recente)."
    ),
)
async def predict_manual(request: Request, body: PredictManualRequest):
    try:
        model_svc = request.app.state.model_service
        result = model_svc.predict(body.prices)

        PREDICTION_COUNT.labels(prediction_type="manual").inc()
        PREDICTION_DURATION.observe(result["inference_time_ms"] / 1000)

        return PredictionResponse(
            symbol=body.symbol or "N/A",
            predicted_price=result["predicted_price"],
            predicted_ratio=result["predicted_ratio"],
            reference_price=result["reference_price"],
            last_known_price=result["last_known_price"],
            expected_change_pct=result["expected_change_pct"],
            prediction_for_date="N/A (data não fornecida)",
            last_data_date="N/A (data não fornecida)",
            inference_time_ms=result["inference_time_ms"],
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Error in predict_manual")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


# ── POST /predict/live ─────────────────────────────────────────────────────────

@router.post(
    "/live",
    response_model=PredictionResponse,
    summary="Predição — dados ao vivo (Yahoo Finance)",
    description=(
        "Busca automaticamente os **últimos 60 dias de fechamento** do Yahoo Finance "
        "para o símbolo informado e retorna a previsão do próximo dia útil."
    ),
)
async def predict_live(request: Request, body: PredictLiveRequest):
    try:
        data = fetch_prices(body.symbol)
        model_svc = request.app.state.model_service
        result = model_svc.predict(data["prices"])

        PREDICTION_COUNT.labels(prediction_type="live").inc()
        PREDICTION_DURATION.observe(result["inference_time_ms"] / 1000)

        pred_date = next_business_day(data["last_date"])

        return PredictionResponse(
            symbol=body.symbol,
            predicted_price=result["predicted_price"],
            predicted_ratio=result["predicted_ratio"],
            reference_price=result["reference_price"],
            last_known_price=result["last_known_price"],
            expected_change_pct=result["expected_change_pct"],
            prediction_for_date=pred_date,
            last_data_date=data["last_date"],
            inference_time_ms=result["inference_time_ms"],
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error in predict_live for symbol '%s'", body.symbol)
        raise HTTPException(
            status_code=500,
            detail=f"Live prediction failed for symbol '{body.symbol}': {exc}",
        ) from exc


# ── POST /predict/forecast ─────────────────────────────────────────────────────

@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Previsão multi-step (N dias à frente)",
    description=(
        "Busca os últimos 60 dias de histórico do Yahoo Finance e realiza **previsão "
        "iterativa** para N dias úteis à frente (máximo 30). Cada preço previsto "
        "alimenta a janela do dia seguinte."
    ),
)
async def forecast(request: Request, body: ForecastRequest):
    try:
        data = fetch_prices(body.symbol)
        model_svc = request.app.state.model_service
        raw_forecasts = model_svc.forecast(data["prices"], body.days)

        PREDICTION_COUNT.labels(prediction_type="forecast").inc()

        forecast_days: list[ForecastDay] = []
        current_date = data["last_date"]
        for item in raw_forecasts:
            next_date = next_business_day(current_date)
            forecast_days.append(
                ForecastDay(
                    day=item["day"],
                    date=next_date,
                    predicted_price=item["predicted_price"],
                    expected_change_pct=item["expected_change_pct"],
                )
            )
            current_date = next_date

        return ForecastResponse(
            symbol=body.symbol,
            base_price=round(data["last_price"], 4),
            base_date=data["last_date"],
            forecast_days=body.days,
            forecast=forecast_days,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error in forecast for symbol '%s'", body.symbol)
        raise HTTPException(
            status_code=500,
            detail=f"Forecast failed for symbol '{body.symbol}': {exc}",
        ) from exc
