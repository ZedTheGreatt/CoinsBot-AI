from __future__ import annotations

from config import APP_NAME


def format_signal(
    symbol: str,
    current_price: float,
    next_candle_probabilities: dict[str, float],
    decision_probabilities: dict[str, float],
    final_decision: str,
    confidence: int,
    market_bias: str,
    timeframe: str,
) -> str:

    symbol = symbol.upper()
    timeframe = timeframe.upper()

    return f"""
🚨 CoinsBot AI Signal

🪙 {symbol} • ⏱ {timeframe}
💰 ₱{current_price:,.2f}

🔮 Next Candle Probabilities:
📈 UP / 📉 DOWN / ⚪ SIDEWAYS
 {next_candle_probabilities['UP']:.2f}% / {next_candle_probabilities['DOWN']:.2f}% / {next_candle_probabilities['SIDEWAYS']:.2f}%

⚡ Decision Probabilities:
🟢 BUY / 🔴 SELL / 🟡 HOLD
  {decision_probabilities['BUY']:.2f}% / {decision_probabilities['SELL']:.2f}% / {decision_probabilities['HOLD']:.2f}%

⚡ Final Decision: {final_decision}

🎯 Confidence: {confidence}/100
📊 Market Bias: {market_bias}

🤖 Powered by {APP_NAME}
""".strip()


def startup_banner() -> str:
    return f"🚀 {APP_NAME} Started"