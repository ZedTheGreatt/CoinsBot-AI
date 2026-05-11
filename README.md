# CoinsBot AI v1

Production-oriented local crypto signal engine for BTC/PHP using Coins.ph OHLCV candles, XGBoost next-candle prediction, and separate trading decision logic.

## Setup

```powershell
cd C:\Users\Zedrick\Desktop\CODE\CoinsBot-AI
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.10+ is supported. Python 3.10-3.12 is recommended for the smoothest package compatibility.

## Train Locally

Use mock candles for a local smoke test:

```powershell
.\.venv\Scripts\python.exe run.py train --mock --limit 1000
```

Use live Coins.ph BTC/PHP 1H candles:

```powershell
.\.venv\Scripts\python.exe run.py train --symbol BTCPHP --interval 1h --limit 1500
```

Recommended live training range: `1000` to `5000` candles.

## Predict

Local mock prediction:

```powershell
.\.venv\Scripts\python.exe run.py predict --mock --limit 1000
```

Live Coins.ph prediction:

```powershell
.\.venv\Scripts\python.exe run.py predict --symbol BTCPHP --interval 1h --limit 1500
```

## Architecture

- `data/fetcher.py`: Coins.ph OHLCV API client and local mock candles
- `features/`: indicators and feature builder
- `model/train.py`: XGBoost next-candle training
- `model/predict.py`: inference
- `engine/`: market state, ATR sideways override, decision probabilities
- `output/formatter.py`: signal formatting
- `app/main.py`: CLI entry point

Prediction logic and decision logic are intentionally separate. BUY is not mapped directly from UP; final decisions require trend, RSI, ATR, volume, and no-trade conditions.
