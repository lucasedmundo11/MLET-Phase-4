import os
from datetime import datetime

import psutil
from fastapi import APIRouter, Request

router = APIRouter()


@router.get(
    "/stats",
    summary="Estatísticas em tempo real da API",
    description=(
        "Retorna métricas de utilização de recursos e status do modelo em formato JSON. "
        "Para métricas no formato Prometheus, acesse `/metrics`."
    ),
    tags=["Monitoring"],
)
async def monitoring_stats(request: Request):
    process = psutil.Process()
    mem = process.memory_info()

    model_svc = getattr(request.app.state, "model_service", None)
    model_loaded = model_svc is not None and model_svc.model is not None

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "system": {
            "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
            "memory_used_mb": round(mem.rss / 1024 / 1024, 2),
            "memory_available_mb": round(
                psutil.virtual_memory().available / 1024 / 1024, 2
            ),
            "pid": os.getpid(),
        },
        "model": {
            "loaded": model_loaded,
            "load_time_ms": model_svc.load_time_ms if model_loaded else None,
        },
        "monitoring_endpoints": {
            "prometheus_metrics": "/metrics",
            "model_info": "/monitoring/model/info",
        },
    }


@router.get(
    "/model/info",
    summary="Metadados e performance do modelo LSTM",
    description="Retorna arquitetura, hiperparâmetros, métricas de teste e informações de treinamento.",
    tags=["Monitoring"],
)
async def model_info(request: Request):
    model_svc = request.app.state.model_service
    meta = model_svc.metadata

    return {
        "symbol": meta.get("symbol"),
        "architecture": meta.get("architecture"),
        "look_back_days": meta.get("look_back"),
        "normalization": meta.get("normalization"),
        "target_description": meta.get("target"),
        "reconstruction": meta.get("reconstruction"),
        "training_period": f"{meta.get('start_date')} → {meta.get('end_date')}",
        "data_splits": {
            "train": meta.get("train_split"),
            "validation": meta.get("val_split"),
            "test": meta.get("test_split"),
        },
        "hyperparameters": {
            "optimizer": meta.get("optimizer"),
            "loss_function": meta.get("loss_function"),
            "batch_size": meta.get("batch_size"),
            "epochs_total": meta.get("epochs_total"),
            "epochs_trained": meta.get("epochs_trained"),
        },
        "metrics_test": meta.get("metrics_test"),
        "created_at": meta.get("created_at"),
        "tf_version": meta.get("tf_version"),
        "model_load_time_ms": model_svc.load_time_ms,
    }
