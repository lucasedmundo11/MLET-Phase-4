import json
import logging
import time

import numpy as np

from app.config import LOOK_BACK

logger = logging.getLogger(__name__)


class ModelService:
    """
    Serviço responsável por carregar e servir o modelo LSTM para inferência.

    Normalização por janela (per-window):
        X[i] = prices[i-60:i] / prices[i-60]   → começa em 1.0
        y[i] = prices[i] / prices[i-60]         → ratio ~1.0
        Reconstrução: pred_price = y_pred * ref_price
    """

    def __init__(self, model_path: str, metadata_path: str) -> None:
        self.model = None
        self.metadata: dict = {}
        self.load_time_ms: float = 0.0
        self._load(model_path, metadata_path)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load(self, model_path: str, metadata_path: str) -> None:
        from tensorflow.keras.models import load_model as keras_load_model  # lazy import
        from app.middleware.metrics import MODEL_LOAD_SUCCESS

        t0 = time.time()
        try:
            self.model = keras_load_model(model_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.load_time_ms = round((time.time() - t0) * 1000, 2)
            MODEL_LOAD_SUCCESS.set(1)
            logger.info("Model loaded in %.0f ms from '%s'", self.load_time_ms, model_path)
        except Exception as exc:
            MODEL_LOAD_SUCCESS.set(0)
            logger.error("Failed to load model from '%s': %s", model_path, exc)
            raise

    # ── Single-step prediction ─────────────────────────────────────────────────

    def predict(self, prices: list[float]) -> dict:
        """
        Prediz o preço de fechamento do próximo dia.

        Args:
            prices: lista de exatamente LOOK_BACK preços de fechamento,
                    ordenada do mais antigo ao mais recente.

        Returns:
            Dicionário com preço previsto e métricas auxiliares.
        """
        if len(prices) != LOOK_BACK:
            raise ValueError(
                f"Expected exactly {LOOK_BACK} prices, got {len(prices)}."
            )

        arr = np.array(prices, dtype=np.float64)
        ref_price = arr[0]  # primeiro elemento da janela → divisor

        # Normalização per-window
        X = (arr / ref_price).reshape(1, LOOK_BACK, 1)

        t0 = time.time()
        ratio = float(self.model.predict(X, verbose=0)[0, 0])
        inference_ms = round((time.time() - t0) * 1000, 2)

        pred_price = ratio * ref_price
        last_price = float(prices[-1])
        change_pct = ((pred_price / last_price) - 1.0) * 100.0

        return {
            "predicted_ratio": round(ratio, 6),
            "predicted_price": round(pred_price, 4),
            "reference_price": round(float(ref_price), 4),
            "last_known_price": round(last_price, 4),
            "expected_change_pct": round(change_pct, 4),
            "inference_time_ms": inference_ms,
        }

    # ── Multi-step forecast ────────────────────────────────────────────────────

    def forecast(self, prices: list[float], days: int) -> list[dict]:
        """
        Previsão iterativa de múltiplos dias usando janela deslizante.

        Cada preço previsto alimenta a janela da próxima iteração.

        Args:
            prices: histórico de preços (precisa ter pelo menos LOOK_BACK valores).
            days:   número de dias à frente a prever.

        Returns:
            Lista de dicts com {day, predicted_price, expected_change_pct}.
        """
        if len(prices) < LOOK_BACK:
            raise ValueError(
                f"Need at least {LOOK_BACK} historical prices, got {len(prices)}."
            )

        # Trabalha com os últimos LOOK_BACK preços
        window = list(prices[-LOOK_BACK:])
        results = []

        for day_idx in range(1, days + 1):
            pred = self.predict(window)
            results.append(
                {
                    "day": day_idx,
                    "predicted_price": pred["predicted_price"],
                    "expected_change_pct": pred["expected_change_pct"],
                }
            )
            # Desliza a janela: descarta o mais antigo, acrescenta a previsão
            window = window[1:] + [pred["predicted_price"]]

        return results
