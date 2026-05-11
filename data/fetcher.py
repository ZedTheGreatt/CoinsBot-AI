from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

from config import COINS_KLINES_URL, COINS_MAX_LIMIT, DEFAULT_INTERVAL, DEFAULT_SYMBOL, MAX_CANDLES


OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class CandleRequest:
    symbol: str = DEFAULT_SYMBOL
    interval: str = DEFAULT_INTERVAL
    limit: int = 1500
    timeout: int = 20
    pause_seconds: float = 0.2


class CoinsPHClient:
    """Coins.ph market-data client for public OHLCV candles."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def fetch_ohlcv(self, request: CandleRequest) -> pd.DataFrame:
        if not 1 <= request.limit <= MAX_CANDLES:
            raise ValueError(f"limit must be between 1 and {MAX_CANDLES}")

        candles: list[list[Any]] = []
        remaining = request.limit
        end_time: int | None = None

        while remaining > 0:
            batch_size = min(remaining, COINS_MAX_LIMIT)
            params: dict[str, Any] = {
                "symbol": request.symbol,
                "interval": request.interval,
                "limit": batch_size,
            }
            if end_time is not None:
                params["endTime"] = end_time

            response = self.session.get(COINS_KLINES_URL, params=params, timeout=request.timeout)
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("data", payload) if isinstance(payload, dict) else payload
            if not batch:
                break

            candles.extend(batch)
            remaining -= len(batch)
            oldest_open_time = int(batch[0][0])
            end_time = oldest_open_time - 1

            if len(batch) < batch_size:
                break
            time.sleep(request.pause_seconds)

        if not candles:
            raise RuntimeError("Coins.ph returned no candles")

        return normalize_ohlcv(candles).tail(request.limit).reset_index(drop=True)


def normalize_ohlcv(raw_candles: list[list[Any]]) -> pd.DataFrame:
    rows = []
    for candle in raw_candles:
        if len(candle) < 6:
            continue
        rows.append(candle[:6])

    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    for column in OHLCV_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return (
        df.dropna()
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_ohlcv(symbol: str = DEFAULT_SYMBOL, interval: str = DEFAULT_INTERVAL, limit: int = 1500) -> pd.DataFrame:
    return CoinsPHClient().fetch_ohlcv(CandleRequest(symbol=symbol, interval=interval, limit=limit))


def make_mock_ohlcv(limit: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Deterministic local data for smoke tests when live API/certificates are unavailable."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=limit, freq="h")

    drift = rng.normal(120, 900, size=limit)
    cycles = np.sin(np.linspace(0, 10 * np.pi, limit)) * 2200
    close = 4_800_000 + np.cumsum(drift + cycles)
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 1800, size=limit)
    spread = rng.uniform(1500, 9500, size=limit)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(mean=5.7, sigma=0.45, size=limit)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
