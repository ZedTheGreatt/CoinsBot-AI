from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketState:
    bias: str
    is_sideways: bool
    bullish_trend: bool
    bearish_trend: bool
    bullish_rsi: bool
    bearish_rsi: bool
    volume_spike: bool
    atr_pct: float


def analyze_market_state(row: pd.Series) -> MarketState:
    bullish_trend = bool(row["ema_50"] > row["ema_200"])
    bearish_trend = bool(row["ema_50"] < row["ema_200"])
    bullish_rsi = bool(row["rsi_14"] > 55)
    bearish_rsi = bool(row["rsi_14"] < 45)
    volume_spike = bool(row.get("volume_ratio_20", 1.0) >= 1.5 or row.get("volume_change", 0.0) >= 0.50)
    is_sideways = bool(row.get("is_low_atr", False))

    if is_sideways:
        bias = "NEUTRAL"
    elif bullish_trend and bullish_rsi:
        bias = "BULLISH"
    elif bearish_trend and bearish_rsi:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return MarketState(
        bias=bias,
        is_sideways=is_sideways,
        bullish_trend=bullish_trend,
        bearish_trend=bearish_trend,
        bullish_rsi=bullish_rsi,
        bearish_rsi=bearish_rsi,
        volume_spike=volume_spike,
        atr_pct=float(row["atr_pct"]),
    )


def apply_market_overrides(next_probs: dict[str, float], state: MarketState) -> dict[str, float]:
    adjusted = {key: float(next_probs.get(key, 0.0)) for key in ["UP", "DOWN", "SIDEWAYS"]}

    if state.is_sideways:
        adjusted["SIDEWAYS"] = max(adjusted["SIDEWAYS"], 70.0)
        adjusted["UP"] *= 0.55
        adjusted["DOWN"] *= 0.55
    else:
        if state.bullish_trend and state.bullish_rsi:
            adjusted["UP"] *= 1.15
            adjusted["DOWN"] *= 0.90
        if state.bearish_trend and state.bearish_rsi:
            adjusted["DOWN"] *= 1.15
            adjusted["UP"] *= 0.90
        if state.volume_spike:
            directional_label = "UP" if adjusted["UP"] >= adjusted["DOWN"] else "DOWN"
            adjusted[directional_label] *= 1.08

    return normalize_percentages(adjusted)


def normalize_percentages(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in values.values())
    if total <= 0:
        return {key: round(100 / len(values), 2) for key in values}
    return {key: round(max(value, 0.0) / total * 100, 2) for key, value in values.items()}
