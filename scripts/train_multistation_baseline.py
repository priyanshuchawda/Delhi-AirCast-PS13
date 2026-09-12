#!/usr/bin/env python3
"""Train and score a global multi-station XGBoost PM2.5 baseline."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DEFAULT_DATASET = Path("data/processed/delhi_multistation_forecast.parquet")
DEFAULT_RUN_DIR = Path("data/runs/multistation_xgboost")
TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-10-01", tz="UTC")


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def _feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    station_categories = sorted(frame["station_id"].astype(str).unique())
    station_codes = {station_id: index for index, station_id in enumerate(station_categories)}
    features = frame.select_dtypes(include="number").copy()
    excluded = {column for column in features.columns if column.startswith("target_")}
    excluded.update({"pm25"})
    features = features.drop(columns=sorted(excluded), errors="ignore")
    station_one_hot = pd.get_dummies(frame["station_id"].astype(str), prefix="station", dtype=float)
    features = pd.concat([features, station_one_hot], axis=1)
    return features, list(features.columns), station_codes


def train(dataset_path: Path, run_dir: Path, horizon: int) -> dict[str, object]:
    target = f"target_pm25_{horizon}h"
    frame = pd.read_parquet(dataset_path).sort_values("timestamp_utc").reset_index(drop=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.dropna(subset=[target, "pm25_current"]).reset_index(drop=True)
    feature_frame, feature_columns, station_codes = _feature_frame(frame)
    feature_frame = feature_frame.astype(float)

    partitions = {
        "train": frame["timestamp_utc"] < TRAIN_END,
        "validation": (frame["timestamp_utc"] >= TRAIN_END) & (frame["timestamp_utc"] < VALIDATION_END),
        "test": frame["timestamp_utc"] >= VALIDATION_END,
    }
    if min(int(mask.sum()) for mask in partitions.values()) == 0:
        raise ValueError("time split produced an empty partition")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        tree_method="hist",
        n_jobs=8,
        random_state=42,
    )
    model.fit(feature_frame.loc[partitions["train"], feature_columns], frame.loc[partitions["train"], target])

    split_metrics: dict[str, dict[str, object]] = {}
    station_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name, mask in partitions.items():
        actual = frame.loc[mask, target]
        predicted = pd.Series(
            model.predict(feature_frame.loc[mask, feature_columns]),
            index=actual.index,
        )
        persistence = frame.loc[mask, "pm25_current"]
        split_metrics[name] = {
            "rows": int(mask.sum()),
            "start_utc": frame.loc[mask, "timestamp_utc"].min().isoformat(),
            "end_utc": frame.loc[mask, "timestamp_utc"].max().isoformat(),
            "persistence": _metrics(actual, persistence),
            "xgboost": _metrics(actual, predicted),
        }
        station_rows: dict[str, dict[str, float]] = {}
        for station_id, station_mask in frame.loc[mask].groupby("station_id").groups.items():
            station_actual = actual.loc[station_mask]
            station_predicted = predicted.loc[station_mask]
            station_persistence = persistence.loc[station_mask]
            station_rows[str(station_id)] = {
                "rows": int(len(station_actual)),
                "persistence_mae": float(mean_absolute_error(station_actual, station_persistence)),
                "xgboost_mae": float(mean_absolute_error(station_actual, station_predicted)),
            }
        station_metrics[name] = station_rows

    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"model_{horizon}h.joblib"
    metrics_path = run_dir / f"metrics_{horizon}h.json"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "station_codes": station_codes,
            "horizon_hours": horizon,
        },
        model_path,
    )
    result: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "horizon_hours": horizon,
        "split_policy": "train < 2025-07-01; validation < 2025-10-01; test >= 2025-10-01",
        "feature_columns": feature_columns,
        "splits": split_metrics,
        "station_metrics": station_metrics,
        "model": str(model_path),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--horizon", type=int, choices=(1, 3, 6, 12, 24), default=1)
    args = parser.parse_args()
    result = train(args.dataset, args.run_dir, args.horizon)
    test = result["splits"]["test"]
    print(
        "Test {h}h MAE: persistence={p:.3f}, xgboost={m:.3f}; rows={rows}".format(
            h=args.horizon,
            p=test["persistence"]["mae"],
            m=test["xgboost"]["mae"],
            rows=test["rows"],
        )
    )


if __name__ == "__main__":
    main()
