from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineer import TARGET_COL, build_feature_table


DEFAULT_MODEL_PATH = "artifacts/daily_return_model.pkl"
DEFAULT_FEATURES_PATH = "artifacts/aapl_features.csv"
DEFAULT_DATA_PATH = "data/AAPL_Stock_Price_Dataset.csv"
DEFAULT_START_DATE = "2000-01-01"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "Date",
        TARGET_COL,
    }
    return [c for c in df.columns if c not in exclude]


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


def evaluate_split(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float((np.sign(y_true) == np.sign(y_pred)).mean()),
    }
    print(f"{name} metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    return metrics


def train_model(
    model_path: str,
    features_path: str = DEFAULT_FEATURES_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    start_date: str = DEFAULT_START_DATE,
) -> Path:
    df = load_feature_data(features_path=features_path, data_path=data_path, start_date=start_date)

    train_mask = df["Date"].dt.year <= 2023
    val_mask = df["Date"].dt.year == 2024

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    if train_df.empty:
        raise ValueError("No training rows found (expected years up to 2023).")
    if val_df.empty:
        raise ValueError("No validation rows found for 2024.")

    feature_cols = get_feature_columns(df)
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    metrics = evaluate_split("Validation (2024)", y_val, val_pred)

    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "target_column": TARGET_COL,
        "train_start_date": start_date,
        "data_path": data_path,
        "features_path": features_path,
        "validation_metrics": metrics,
    }

    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(artifact, f)

    artifact_path = out_path.with_suffix(".artifact.json")
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, default=str)

    print(f"Saved model artifact to {out_path}")
    print(f"Saved artifact metadata to {artifact_path}")
    return out_path


def main() -> None:
    train_model(
        model_path=DEFAULT_MODEL_PATH,
        features_path=DEFAULT_FEATURES_PATH,
        data_path=DEFAULT_DATA_PATH,
        start_date=DEFAULT_START_DATE,
    )


if __name__ == "__main__":
    main()
