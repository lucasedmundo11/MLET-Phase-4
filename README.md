# Tech Challenge Fase 4 — LSTM Stock Price Predictor

> **Pos-Tech FIAP — Machine Learning Engineering**
> Modelo LSTM para previsão do preço de fechamento de ações + API RESTful com monitoramento em produção.

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Sistema](#arquitetura-do-sistema)
3. [Estrutura do Projeto](#estrutura-do-projeto)
4. [Modelo LSTM](#modelo-lstm)
5. [API Endpoints](#api-endpoints)
6. [Como Executar](#como-executar)
7. [Docker](#docker)
8. [Monitoramento](#monitoramento)

---

## Visão Geral

Aplicação completa de Machine Learning Engineering cobrindo toda a pipeline:

| Etapa | Descrição |
|-------|-----------|
| Coleta | Preços históricos via **yfinance** (PETR4.SA, 2018–2024) |
| Pré-processamento | Feature engineering, normalização por janela, splits 70/15/15 |
| EDA | Análise estatística, volatilidade, sazonalidade, ACF/PACF |
| Modelagem | LSTM 2 camadas com regularização L2, Dropout, BatchNorm |
| Avaliação | MAE, RMSE, MAPE, R² nos conjuntos de treino/validação/teste |
| Deploy | **FastAPI** containerizada com **Docker** |
| Monitoramento | **Prometheus** + **Grafana** (latência, throughput, CPU, memória) |

---

## Arquitetura do Sistema

```text
┌─────────────────────────────────────────────────────────────┐
│                        Docker Compose                        │
│                                                             │
│   ┌─────────────┐    /metrics    ┌───────────────────────┐ │
│   │  FastAPI    │◄───────────────│     Prometheus        │ │
│   │ (porta 8000)│               │     (porta 9090)      │ │
│   └──────┬──────┘               └───────────┬───────────┘ │
│          │ LSTM                              │             │
│          │ Inference                         ▼             │
│   ┌──────▼──────┐               ┌───────────────────────┐ │
│   │   Modelo    │               │       Grafana         │ │
│   │  LSTM Keras │               │     (porta 3000)      │ │
│   └─────────────┘               └───────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Estrutura do Projeto

```text
MLET-Phase-4/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── config.py                # Configurações e variáveis de ambiente
│   ├── middleware/
│   │   └── metrics.py           # Middleware Prometheus (latência, CPU, memória)
│   ├── routers/
│   │   ├── health.py            # GET / e GET /health
│   │   ├── predict.py           # POST /predict, /predict/live, /predict/forecast
│   │   └── monitoring.py        # GET /monitoring/stats e /monitoring/model/info
│   ├── services/
│   │   ├── model_service.py     # Carregamento do modelo + inferência LSTM
│   │   └── data_service.py      # Busca de dados via yfinance
│   └── schemas/
│       └── prediction.py        # Modelos Pydantic (request/response)
│
├── models/
│   ├── lstm_petr4_final.keras   # Modelo treinado (TF 2.20)
│   ├── lstm_petr4_best.keras    # Melhor checkpoint durante treino
│   └── model_metadata.json      # Metadados, hiperparâmetros e métricas
│
├── notebooks/
│   └── mvp vPROD.ipynb          # Notebook completo: EDA → Treino → Exportação
│
├── monitoring/
│   ├── prometheus.yml           # Configuração do Prometheus
│   └── grafana/
│       └── provisioning/        # Datasource e dashboard pré-configurados
│
├── docs/
│   └── Pos_Tech - MLET - Tech Challenge Fase 4.pdf
│
├── Dockerfile                   # Multi-stage build (python:3.11-slim)
├── docker-compose.yml           # Stack: API + Prometheus + Grafana
└── requirements.txt
```

---

## Modelo LSTM

### Arquitetura da Rede

```text
Input: [batch, 60, 1]   (60 dias de preços normalizados por janela)
  ↓
LSTM(128, return_sequences=True)  + L2(5e-4) + recurrent_dropout=10%
  ↓
Dropout(25%)
  ↓
LSTM(64, return_sequences=False)  + L2(5e-4) + recurrent_dropout=10%
  ↓
BatchNormalization
  ↓
Dropout(25%)
  ↓
Dense(32, relu)  + L2(5e-4)
  ↓
Dense(1)   →  ratio = Close[t] / Close[t-60]   (~1.0)
```

**Total de parâmetros:** 118.337 treináveis

### Normalização por Janela (per-window normalization)

```python
ref = prices[0]          # primeiro elemento da janela
X   = prices / ref       # janela começa em 1.0
y   = price_next / ref   # target é o ratio (~1.0)

# Reconstrução do preço real:
pred_price = model.predict(X) * ref
```

Essa estratégia elimina _distribution shift_ entre os splits sem usar scaler global, evitando _data leakage_.

### Métricas de Avaliação

| Conjunto | MAE (R$) | RMSE (R$) | MAPE (%) | R² |
|----------|----------|-----------|----------|----|
| Treino | 0.2050 | 0.2906 | 2.65% | 0.9868 |
| Validação | 0.4626 | 0.5980 | 2.93% | 0.9463 |
| **Teste** | **0.5008** | **0.7087** | **1.83%** | **0.9360** |

---

## API Endpoints

### Tabela de Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/` | Informações da API e links |
| `GET` | `/health` | Status de saúde e modelo |
| `GET` | `/docs` | Swagger UI (documentação interativa) |
| `GET` | `/redoc` | ReDoc (documentação alternativa) |
| `GET` | `/metrics` | Métricas Prometheus |
| `POST` | `/predict` | Predição com preços manuais (60 valores) |
| `POST` | `/predict/live` | Predição com busca automática (Yahoo Finance) |
| `POST` | `/predict/forecast` | Forecast multi-step (1–30 dias) |
| `GET` | `/monitoring/stats` | Métricas de sistema em tempo real (JSON) |
| `GET` | `/monitoring/model/info` | Metadados e performance do modelo |

### Predição com dados ao vivo

```bash
curl -X POST "http://localhost:8000/predict/live" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "PETR4.SA"}'
```

**Resposta:**

```json
{
  "symbol": "PETR4.SA",
  "predicted_price": 31.37,
  "predicted_ratio": 1.038158,
  "reference_price": 30.21,
  "last_known_price": 31.29,
  "expected_change_pct": 0.23,
  "prediction_for_date": "2024-07-22",
  "last_data_date": "2024-07-19",
  "inference_time_ms": 45.2,
  "timestamp": "2024-07-19T20:00:00Z"
}
```

### Forecast de 5 dias

```bash
curl -X POST "http://localhost:8000/predict/forecast" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "PETR4.SA", "days": 5}'
```

### Predição com preços manuais

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "PETR4.SA", "prices": [28.1, 28.3, 28.7, ...]}'
```

> Os 60 preços devem estar ordenados do mais antigo ao mais recente.

---

## Como Executar

### Localmente (sem Docker)

Pré-requisitos: Python 3.10+

```bash
# 1. Clone o repositório
git clone <repo-url>
cd MLET-Phase-4

# 2. Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
# macOS Apple Silicon: pip install tensorflow-macos tensorflow-metal

# 4. Inicie a API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse a documentação em: <http://localhost:8000/docs>

---

## Docker

### Apenas a API

```bash
docker build -t petr4-lstm-api .
docker run -p 8000:8000 petr4-lstm-api
```

### Stack completa (API + Prometheus + Grafana)

```bash
# Subir toda a stack em background
docker compose up -d --build

# Verificar status dos containers
docker compose ps

# Acompanhar logs da API
docker compose logs -f api

# Parar a stack
docker compose down
```

**URLs após inicialização:**

| Serviço | URL | Credenciais |
| ------- | --- | ----------- |
| API (Swagger) | <http://localhost:8000/docs> | — |
| API Health | <http://localhost:8000/health> | — |
| Prometheus | <http://localhost:9090> | — |
| Grafana | <http://localhost:3000> | admin / admin123 |

> O Grafana já vem com o datasource do Prometheus e o dashboard **"PETR4 LSTM API — Monitoramento"** pré-configurados.

---

## Monitoramento

### Métricas Prometheus

| Métrica | Tipo | Descrição |
| ------- | ---- | --------- |
| `http_requests_total` | Counter | Total de requisições por método/endpoint/status |
| `http_request_duration_seconds` | Histogram | Latência das requisições |
| `http_active_requests` | Gauge | Requisições simultâneas em andamento |
| `predictions_total` | Counter | Total de predições por tipo (manual/live/forecast) |
| `prediction_duration_seconds` | Histogram | Tempo de inferência do modelo |
| `model_loaded` | Gauge | Status do modelo (1=carregado, 0=falhou) |
| `process_cpu_usage_percent` | Gauge | Uso de CPU pelo processo |
| `process_memory_usage_bytes` | Gauge | Uso de memória RAM |

### Dashboard Grafana

O dashboard pré-configurado exibe:

- Status do modelo em tempo real
- Requisições por segundo
- Latência P95 por endpoint
- Distribuição de predições por tipo
- Uso de CPU e memória ao longo do tempo

### Métricas em JSON (sem Prometheus)

```bash
curl http://localhost:8000/monitoring/stats
```

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `MODEL_PATH` | `models/lstm_petr4_final.keras` | Caminho para o modelo Keras |
| `METADATA_PATH` | `models/model_metadata.json` | Caminho para os metadados |
| `LOOK_BACK` | `60` | Tamanho da janela de histórico (dias) |

---

## Dependências Principais

| Pacote | Versão | Uso |
|--------|--------|-----|
| `fastapi` | ≥0.111 | Framework da API |
| `uvicorn` | ≥0.29 | Servidor ASGI |
| `tensorflow` | ≥2.15 | Inferência do modelo LSTM |
| `pandas` / `numpy` | ≥2.0 / ≥1.24 | Manipulação de dados |
| `yfinance` | ≥0.2.37 | Dados financeiros em tempo real |
| `prometheus-client` | ≥0.20 | Exportação de métricas |
| `psutil` | ≥5.9 | Métricas de sistema |
