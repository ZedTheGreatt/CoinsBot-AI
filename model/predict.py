from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from config import NEXT_CANDLE_MODEL_PATH
from features import FEATURE_COLUMNS


class NextCandlePredictor:
    def __init__(self, model_path: Path = NEXT_CANDLE_MODEL_PATH) -> None:
        self.model_path = model_path
        self.artifact = self._load_artifact()

    def predict_probabilities(self, feature_row: pd.Series) -> dict[str, float]:
        feature_columns = self.artifact.get("feature_columns", FEATURE_COLUMNS)
        X = pd.DataFrame([feature_row[feature_columns].to_dict()], columns=feature_columns)
        raw_probabilities = self.artifact["model"].predict_proba(X)[0]
        labels = self.artifact["labels"]

        probabilities = {labels[index]: round(float(value) * 100, 2) for index, value in enumerate(raw_probabilities)}
        for label in ["UP", "DOWN", "SIDEWAYS"]:
            probabilities.setdefault(label, 0.0)
        return probabilities

    def _load_artifact(self) -> dict:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run `python run.py train --mock` first.")
        return joblib.load(self.model_path)


def predict_next_candle(feature_row: pd.Series, model_path: Path = NEXT_CANDLE_MODEL_PATH) -> dict[str, float]:
    return NextCandlePredictor(model_path=model_path).predict_probabilities(feature_row)
