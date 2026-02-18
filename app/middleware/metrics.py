import time

import psutil
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# ── Prometheus Metrics Definitions ────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisições HTTP recebidas",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Duração das requisições HTTP em segundos",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total de predições realizadas pela API",
    ["prediction_type"],  # "manual", "live", "forecast"
)

PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds",
    "Duração das predições do modelo LSTM em segundos",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

MODEL_LOAD_SUCCESS = Gauge(
    "model_loaded",
    "Indica se o modelo LSTM está carregado (1 = sim, 0 = não)",
)

CPU_USAGE = Gauge(
    "process_cpu_usage_percent",
    "Uso de CPU pelo processo da API (%)",
)

MEMORY_USAGE_BYTES = Gauge(
    "process_memory_usage_bytes",
    "Uso de memória RAM pelo processo da API (bytes)",
)

ACTIVE_REQUESTS = Gauge(
    "http_active_requests",
    "Número de requisições HTTP em andamento",
)


# ── Middleware ─────────────────────────────────────────────────────────────────

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware que intercepta todas as requisições para:
    - Registrar contagem e duração por endpoint/método/status.
    - Manter métricas de utilização de CPU e memória atualizadas.
    - Rastrear requisições ativas simultâneas.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method

        ACTIVE_REQUESTS.inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
        finally:
            ACTIVE_REQUESTS.dec()

        duration = time.perf_counter() - start
        status = str(response.status_code)

        REQUEST_COUNT.labels(method=method, endpoint=path, status_code=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)

        # Atualiza métricas de sistema (leve, sem bloqueio de I/O)
        try:
            CPU_USAGE.set(psutil.cpu_percent(interval=None))
            MEMORY_USAGE_BYTES.set(psutil.Process().memory_info().rss)
        except Exception:
            pass

        return response
