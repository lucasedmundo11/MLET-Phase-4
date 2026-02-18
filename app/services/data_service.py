import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from app.config import LOOK_BACK

logger = logging.getLogger(__name__)


def fetch_prices(symbol: str, n_prices: int = LOOK_BACK) -> dict:
    """
    Busca os últimos n_prices preços de fechamento de um ativo via Yahoo Finance.

    Args:
        symbol:   Código do ativo (ex: 'PETR4.SA', 'VALE3.SA', 'AAPL').
        n_prices: Número de dias úteis de histórico necessários.

    Returns:
        {
            "prices":     [float, ...],     # preços ordenados do mais antigo ao mais recente
            "dates":      ["YYYY-MM-DD", ...],
            "symbol":     str,
            "last_date":  str,
            "last_price": float,
        }

    Raises:
        ValueError: se não houver dados suficientes para o símbolo.
    """
    # Busca com buffer para cobrir fins de semana e feriados
    end = datetime.today()
    start = end - timedelta(days=n_prices * 4)

    logger.info("Fetching %d prices for '%s' from %s", n_prices, symbol, start.date())

    df = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(f"No market data found for symbol '{symbol}'.")

    # yfinance pode retornar MultiIndex de colunas
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    close = df["Close"].dropna()

    if len(close) < n_prices:
        raise ValueError(
            f"Insufficient data for '{symbol}': "
            f"found {len(close)} trading days, need {n_prices}."
        )

    last_n = close.iloc[-n_prices:]
    prices = [float(p) for p in last_n.values]
    dates = [str(d.date()) for d in last_n.index]

    return {
        "prices": prices,
        "dates": dates,
        "symbol": symbol,
        "last_date": dates[-1],
        "last_price": prices[-1],
    }


def next_business_day(from_date_str: str) -> str:
    """Retorna a próxima data útil após a data fornecida (formato 'YYYY-MM-DD')."""
    ts = pd.Timestamp(from_date_str)
    next_bd = ts + pd.tseries.offsets.BDay(1)
    return str(next_bd.date())
