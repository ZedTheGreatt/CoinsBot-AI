from __future__ import annotations

import argparse
import sys

from config import DEFAULT_CANDLE_LIMIT, DEFAULT_INTERVAL, DEFAULT_SYMBOL, MAX_CANDLES, MIN_CANDLES
from data.fetcher import fetch_ohlcv, make_mock_ohlcv
from engine import analyze_market_state, apply_market_overrides, decide
from features import latest_features
from model.predict import predict_next_candle
from model.train import train_next_candle_model
from output.formatter import format_signal


def generate_signal(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_CANDLE_LIMIT,
    use_live: bool = True,
) -> str:
    raw_candles = fetch_ohlcv(symbol=symbol, interval=interval, limit=limit) if use_live else make_mock_ohlcv(limit=limit)
    latest_row = latest_features(raw_candles)

    model_probabilities = predict_next_candle(latest_row)
    market_state = analyze_market_state(latest_row)
    next_candle_probabilities = apply_market_overrides(model_probabilities, market_state)
    decision = decide(next_candle_probabilities, market_state)

    return format_signal(
        symbol=symbol,
        current_price=float(latest_row["close"]),
        next_candle_probabilities=next_candle_probabilities,
        decision_probabilities=decision.decision_probabilities,
        final_decision=decision.final_decision,
        confidence=decision.confidence,
        market_bias=decision.market_bias,
        timeframe=interval,
    )


def train(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_CANDLE_LIMIT,
    use_live: bool = True,
) -> None:
    result = train_next_candle_model(symbol=symbol, interval=interval, limit=limit, use_live=use_live)
    print(f"Saved next-candle model: {result.model_path}")
    print(f"Rows: {result.rows}")
    print(f"Validation accuracy: {result.accuracy:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoinsBot AI v1 - BTC/PHP 1H ML signal engine")
    parser.add_argument("command", choices=["train", "predict"], help="Train a model or generate a signal")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Coins.ph symbol, default BTCPHP")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Candle interval, default 1h")
    parser.add_argument("--limit", type=int, default=DEFAULT_CANDLE_LIMIT, help=f"Candle count, {MIN_CANDLES}-{MAX_CANDLES} recommended")
    parser.add_argument("--mock", action="store_true", help="Use deterministic local candles instead of Coins.ph API")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.command == "train":
        train(symbol=args.symbol, interval=args.interval, limit=args.limit, use_live=not args.mock)
    else:
        print(generate_signal(symbol=args.symbol, interval=args.interval, limit=args.limit, use_live=not args.mock))


if __name__ == "__main__":
    main()
