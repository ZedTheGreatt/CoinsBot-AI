from __future__ import annotations

from pathlib import Path


APP_NAME = "CoinsBot AI v1"

DEFAULT_SYMBOL = "BTCPHP"
DEFAULT_INTERVAL = "1h"
DEFAULT_CANDLE_LIMIT = 1500
MIN_CANDLES = 1000
MAX_CANDLES = 5000

COINS_KLINES_URL = "https://api.pro.coins.ph/openapi/quote/v1/klines"
COINS_MAX_LIMIT = 1000

MODEL_DIR = Path("model/artifacts")
NEXT_CANDLE_MODEL_PATH = MODEL_DIR / "next_candle_xgb.joblib"
DATASET_PATH = Path("data/dataset.csv")

RANDOM_STATE = 42
