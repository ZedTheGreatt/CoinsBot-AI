from __future__ import annotations

from config import APP_NAME


def format_signal(
    next_candle_probabilities: dict[str, float],
    decision_probabilities: dict[str, float],
    final_decision: str,
    confidence: int,
    market_bias: str,
    timeframe: str,
) -> str:
    return f"""🚨 CoinsBot AI Signal

🔮 Next Candle Probabilities:
📈 UP: {next_candle_probabilities['UP']:.2f}%
📉 DOWN: {next_candle_probabilities['DOWN']:.2f}%
⚪ SIDEWAYS: {next_candle_probabilities['SIDEWAYS']:.2f}%

⚡ Decision Probabilities:
🟢 BUY: {decision_probabilities['BUY']:.2f}%
🔴 SELL: {decision_probabilities['SELL']:.2f}%
⚪ HOLD: {decision_probabilities['HOLD']:.2f}%

⚡ Final Decision: {final_decision}

🎯 Confidence: {confidence}/100
📊 Market Bias: {market_bias}
⏱ Timeframe: {timeframe.upper()}"""


def startup_banner() -> str:
    return APP_NAME
