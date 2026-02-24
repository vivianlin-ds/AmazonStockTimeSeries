from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineer import TARGET_COL, build_feature_table

DEFAULT_MODEL_PATH = "artifacts/daily_return_model.pkl"
DEFAULT_FEATURES_PATH = "artifacts/aapl_features.csv"
DEFAULT_DATA_PATH = "data/AAPL_Stock_Price_Dataset.csv"
DEFAULT_START_DATE = "2000-01-01"
DEFAULT_PREDICT_YEAR = 2025
DEFAULT_OUTPUT_PATH = "artifacts/2025_daily_return_predictions.csv"


def load_artifact(model_path: str) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_if_available(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns or df[TARGET_COL].isna().all():
        return

    y_true = df[TARGET_COL]
    y_pred = df["Predicted_Daily_Return"]
    print("2025 metrics:")
    print(f"  mae: {mean_absolute_error(y_true, y_pred):.6f}")
    print(f"  rmse: {np.sqrt(mean_squared_error(y_true, y_pred)):.6f}")
    print(f"  r2: {r2_score(y_true, y_pred):.6f}")
    print(f"  directional_accuracy: {(np.sign(y_true) == np.sign(y_pred)).mean():.6f}")


def load_feature_data(
    features_path: str = DEFAULT_FEATURES_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    start_date: str = DEFAULT_START_DATE,
) -> pd.DataFrame:
    features_file = Path(features_path)
    if features_file.exists():
        df = pd.read_csv(features_file, parse_dates=["Date"])
        return df.sort_values("Date").reset_index(drop=True)

    print(f"Feature file not found at {features_file}; rebuilding from raw data.")
    return build_feature_table(csv_path=data_path, start_date=start_date)


def run_inference(
    model_path: str,
    output_path: str,
    predict_year: int,
    features_path: str = DEFAULT_FEATURES_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    start_date: str = DEFAULT_START_DATE,
) -> Path:
    artifact = load_artifact(model_path)
    feature_cols = artifact["feature_columns"]
    model = artifact["model"]

    df = load_feature_data(features_path=features_path, data_path=data_path, start_date=start_date)
    pred_df = df[df["Date"].dt.year == predict_year].copy()
    if pred_df.empty:
        raise ValueError(f"No rows found for prediction year {predict_year}.")

    missing_cols = [c for c in feature_cols if c not in pred_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns for inference: {missing_cols}")

    pred_df["Predicted_Daily_Return"] = model.predict(pred_df[feature_cols])
    keep_cols = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        TARGET_COL,
        "Predicted_Daily_Return",
    ]
    keep_cols = [c for c in keep_cols if c in pred_df.columns]
    output_df = pred_df[keep_cols].copy()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)

    print(f"Saved {len(output_df)} predictions to {out_path}")
    score_if_available(output_df)
    return out_path


def main() -> None:
    run_inference(
        model_path=DEFAULT_MODEL_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        predict_year=DEFAULT_PREDICT_YEAR,
        features_path=DEFAULT_FEATURES_PATH,
        data_path=DEFAULT_DATA_PATH,
        start_date=DEFAULT_START_DATE,
    )


if __name__ == "__main__":
    main()
