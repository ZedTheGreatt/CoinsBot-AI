from __future__ import annotations

from data.fetcher import fetch_ohlcv, make_mock_ohlcv


def get_live_btc(symbol: str = "BTCPHP", interval: str = "1h", limit: int = 1500):
    return fetch_ohlcv(symbol=symbol, interval=interval, limit=limit)


def get_mock_btc(limit: int = 1500):
    return make_mock_ohlcv(limit=limit)
