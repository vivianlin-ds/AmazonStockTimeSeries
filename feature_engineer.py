from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = "data/AAPL_Stock_Price_Dataset.csv"
DEFAULT_START_DATE = "2000-01-01"
TARGET_COL = "Daily_Return"


def load_base_data(csv_path: str = DEFAULT_DATA_PATH, start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[df["Date"] >= pd.Timestamp(start_date)].copy()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("Date").reset_index(drop=True)

    # Use the dataset target when available; fall back to a computed pct change.
    if TARGET_COL not in data.columns:
        data[TARGET_COL] = data["Close"].pct_change() * 100.0

    # Moving averages (shifted by 1 day to avoid using same-day close in features).
    shifted_close = data["Close"].shift(1)
    data["MA_7"] = shifted_close.rolling(7).mean()
    data["MA_30"] = shifted_close.rolling(30).mean()
    data["MA_90"] = shifted_close.rolling(90).mean()

    # 7-day realized volatility of daily returns (percent units), using prior days only.
    data["Volatility_7D"] = data[TARGET_COL].shift(1).rolling(7).std()

    # Requested lag features.
    data["Close_t-1"] = data["Close"].shift(1)
    data["Close_t-2"] = data["Close"].shift(2)
    data["Close_t-5"] = data["Close"].shift(5)
    data["Daily_Return_t-1"] = data[TARGET_COL].shift(1)
    data["Daily_Return_t-2"] = data[TARGET_COL].shift(2)
    data["Daily_Return_t-5"] = data[TARGET_COL].shift(5)

    data["Volume_t-1"] = data["Volume"].shift(1)

    keep_cols = [
        "Date",
        TARGET_COL,
        "MA_7", "MA_30", "MA_90",
        "Volatility_7D",
        "Close_t-1", "Close_t-2", "Close_t-5",
        "Daily_Return_t-1", "Daily_Return_t-2", "Daily_Return_t-5",
        "Volume_t-1",
    ]
    feature_df = data[keep_cols].dropna().reset_index(drop=True)
    return feature_df


def build_feature_table(
    csv_path: str = DEFAULT_DATA_PATH,
    start_date: str = DEFAULT_START_DATE,
) -> pd.DataFrame:
    base = load_base_data(csv_path=csv_path, start_date=start_date)
    return engineer_features(base)


def main() -> None:
    output_path = "artifacts/aapl_features.csv"
    features = build_feature_table(csv_path=DEFAULT_DATA_PATH, start_date=DEFAULT_START_DATE)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"Saved {len(features)} rows to {out_path}")


if __name__ == "__main__":
    main()
