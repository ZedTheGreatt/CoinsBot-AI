from __future__ import annotations

import pandas as pd

try:
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator, MACD
    from ta.volatility import AverageTrueRange
except ModuleNotFoundError:
    RSIIndicator = None
    EMAIndicator = None
    MACD = None
    AverageTrueRange = None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]

    if RSIIndicator and EMAIndicator and MACD and AverageTrueRange:
        data["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
        data["ema_50"] = EMAIndicator(close=close, window=50).ema_indicator()
        data["ema_200"] = EMAIndicator(close=close, window=200).ema_indicator()
        data["macd_hist"] = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
        data["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    else:
        data["rsi_14"] = rsi(close, 14)
        data["ema_50"] = ema(close, 50)
        data["ema_200"] = ema(close, 200)
        macd_line = ema(close, 12) - ema(close, 26)
        data["macd_hist"] = macd_line - ema(macd_line, 9)
        data["atr_14"] = atr(high, low, close, 14)

    data["atr_pct"] = data["atr_14"] / data["close"]
    return data


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    relative_strength = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + relative_strength))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False).mean()
