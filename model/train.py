from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from config import (
    DATASET_PATH,
    DEFAULT_CANDLE_LIMIT,
    DEFAULT_INTERVAL,
    DEFAULT_SYMBOL,
    MIN_CANDLES,
    NEXT_CANDLE_MODEL_PATH,
    RANDOM_STATE,
)
from data.fetcher import fetch_ohlcv, make_mock_ohlcv
from features import FEATURE_COLUMNS, NEXT_CANDLE_LABELS, build_feature_frame


@dataclass(frozen=True)
class TrainingResult:
    accuracy: float
    rows: int
    model_path: str


def train_next_candle_model(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_CANDLE_LIMIT,
    use_live: bool = True,
) -> TrainingResult:
    if limit < MIN_CANDLES:
        raise ValueError(f"Use at least {MIN_CANDLES} candles for training.")

    raw = fetch_ohlcv(symbol=symbol, interval=interval, limit=limit) if use_live else make_mock_ohlcv(limit=limit)
    dataset = build_feature_frame(raw, with_labels=True)
    if len(dataset) < 300:
        raise ValueError("Not enough rows after indicator warmup. Increase --limit.")

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_PATH, index=False)

    X = dataset[FEATURE_COLUMNS]
    y = dataset["next_candle_label"]

    split_index = int(len(dataset) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = build_xgboost_classifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[NEXT_CANDLE_LABELS[index] for index in [0, 1, 2]],
        zero_division=0,
    )
    print(report)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "labels": NEXT_CANDLE_LABELS,
        "symbol": symbol,
        "interval": interval,
        "rows": len(dataset),
        "accuracy": accuracy,
    }
    NEXT_CANDLE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, NEXT_CANDLE_MODEL_PATH)

    return TrainingResult(accuracy=accuracy, rows=len(dataset), model_path=str(NEXT_CANDLE_MODEL_PATH))


def build_xgboost_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=450,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=2,
        reg_lambda=1.5,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )


def cross_validate_next_candle_model(dataset: pd.DataFrame, splits: int = 4) -> list[float]:
    scores: list[float] = []
    tscv = TimeSeriesSplit(n_splits=splits)
    X = dataset[FEATURE_COLUMNS]
    y = dataset["next_candle_label"]

    for train_index, test_index in tscv.split(X):
        model = build_xgboost_classifier()
        model.fit(X.iloc[train_index], y.iloc[train_index])
        scores.append(float(model.score(X.iloc[test_index], y.iloc[test_index])))

    return scores
