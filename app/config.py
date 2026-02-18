import os

# ── Model paths (override via environment variables for Docker) ────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/lstm_petr4_final.keras")
METADATA_PATH = os.getenv("METADATA_PATH", "models/model_metadata.json")

# ── Model hyperparameters (must match training) ────────────────────────────────
LOOK_BACK: int = int(os.getenv("LOOK_BACK", "60"))

# ── API settings ───────────────────────────────────────────────────────────────
API_TITLE = "PETR4.SA LSTM Stock Price Predictor"
API_DESCRIPTION = (
    "API RESTful para previsão do preço de fechamento de ações utilizando "
    "um modelo LSTM treinado com dados históricos. "
    "Empresa padrão: Petrobras (PETR4.SA, B3)."
)
API_VERSION = "1.0.0"
