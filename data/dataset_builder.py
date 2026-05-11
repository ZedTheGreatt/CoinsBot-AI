from __future__ import annotations

import pandas as pd

from features import build_feature_frame


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return build_feature_frame(df, with_labels=True)
