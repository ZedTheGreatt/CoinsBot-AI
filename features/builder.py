from __future__ import annotations

import numpy as np
import pandas as pd

from features.indicators import add_technical_indicators


NEXT_CANDLE_LABELS = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}
NEXT_CANDLE_IDS = {label: index for index, label in NEXT_CANDLE_LABELS.items()}

FEATURE_COLUMNS = [
    "rsi_14",
    "ema_50",
    "ema_200",
    "ema_spread_pct",
    "macd_hist",
    "atr_14",
    "atr_pct",
    "volume_change",
    "volume_ratio_20",
    "return_1",
    "return_3",
    "return_6",
    "momentum_3",
    "momentum_6",
    "range_pct",
]


def build_feature_frame(df: pd.DataFrame, with_labels: bool = False) -> pd.DataFrame:
    data = add_technical_indicators(df)

    data["return_1"] = data["close"].pct_change(1)
    data["return_3"] = data["close"].pct_change(3)
    data["return_6"] = data["close"].pct_change(6)
    data["momentum_3"] = data["close"] / data["close"].shift(3) - 1
    data["momentum_6"] = data["close"] / data["close"].shift(6) - 1
    data["range_pct"] = (data["high"] - data["low"]) / data["close"]
    data["ema_spread_pct"] = (data["ema_50"] - data["ema_200"]) / data["close"]
    data["volume_change"] = data["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    data["volume_ratio_20"] = data["volume"] / data["volume"].rolling(20).mean()
    data["atr_low_threshold"] = data["atr_pct"].rolling(200, min_periods=80).quantile(0.25)
    data["is_low_atr"] = data["atr_pct"] <= data["atr_low_threshold"]

    if with_labels:
        data["future_return"] = data["close"].shift(-1) / data["close"] - 1
        data["next_candle_label"] = build_next_candle_labels(data)
        data["decision_label"] = build_decision_labels(data)

    return data.dropna().reset_index(drop=True)


def build_next_candle_labels(df: pd.DataFrame) -> pd.Series:
    movement_threshold = np.maximum(df["atr_pct"] * 0.15, 0.00035)
    labels = np.select(
        [
            df["future_return"] > movement_threshold,
            df["future_return"] < -movement_threshold,
        ],
        [
            NEXT_CANDLE_IDS["UP"],
            NEXT_CANDLE_IDS["DOWN"],
        ],
        default=NEXT_CANDLE_IDS["SIDEWAYS"],
    )
    return pd.Series(labels, index=df.index, dtype="int64")


def build_decision_labels(df: pd.DataFrame) -> pd.Series:
    """Optional BUY/SELL/HOLD target structure for a future decision model."""
    bullish_setup = (df["ema_50"] > df["ema_200"]) & (df["rsi_14"] > 55)
    bearish_setup = (df["ema_50"] < df["ema_200"]) & (df["rsi_14"] < 45)
    tradable_move = df["future_return"].abs() > np.maximum(df["atr_pct"] * 0.25, 0.0007)
    low_atr = df["is_low_atr"]

    labels = np.select(
        [
            bullish_setup & tradable_move & (df["future_return"] > 0) & ~low_atr,
            bearish_setup & tradable_move & (df["future_return"] < 0) & ~low_atr,
        ],
        ["BUY", "SELL"],
        default="HOLD",
    )
    return pd.Series(labels, index=df.index)


def latest_features(df: pd.DataFrame) -> pd.Series:
    features = build_feature_frame(df, with_labels=False)
    if features.empty:
        raise ValueError("Not enough candles to build indicators; provide at least 250 candles.")
    return features.iloc[-1]
