"""Train and evaluate persistence plus a tree baseline on a time split."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATASET = Path("data/processed/site_105_next_hour_forecast.parquet")
DEFAULT_RUN_DIR = Path("data/runs/site_105_next_hour_baseline")


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def train(dataset_path: Path, run_dir: Path) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path).sort_values("timestamp_utc")
    target = "target_pm25_next_hour"
    excluded = {
        target,
        "timestamp_utc",
        "timestamp_ist",
        "target_timestamp_utc",
        "station_id",
        "station_name",
        "source",
        "source_resource_id",
    }
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded and frame[column].notna().mean() >= 0.80
    ]
    feature_columns = numeric_columns
    frame = frame.dropna(subset=[target, *feature_columns]).reset_index(drop=True)

    train_end = pd.Timestamp("2025-07-01", tz="UTC")
    validation_end = pd.Timestamp("2025-10-01", tz="UTC")
    train_frame = frame[frame["timestamp_utc"] < train_end]
    validation_frame = frame[
        (frame["timestamp_utc"] >= train_end) & (frame["timestamp_utc"] < validation_end)
    ]
    test_frame = frame[frame["timestamp_utc"] >= validation_end]
    if min(len(train_frame), len(validation_frame), len(test_frame)) == 0:
        raise ValueError("Time split produced an empty partition")

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train_frame[feature_columns], train_frame[target])

    split_metrics: dict[str, dict[str, object]] = {}
    for name, partition in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        actual = partition[target]
        persistence = partition["pm25"]
        predicted = model.predict(partition[feature_columns])
        split_metrics[name] = {
            "rows": int(len(partition)),
            "start_utc": partition["timestamp_utc"].min().isoformat(),
            "end_utc": partition["timestamp_utc"].max().isoformat(),
            "persistence": _metrics(actual, persistence),
            "hist_gradient_boosting": _metrics(actual, predicted),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.joblib"
    metrics_path = run_dir / "metrics.json"
    joblib.dump({"model": model, "feature_columns": feature_columns}, model_path)
    result: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "forecast_horizon_hours": 1,
        "split_policy": "train < 2025-07-01; validation < 2025-10-01; test >= 2025-10-01",
        "feature_columns": feature_columns,
        "splits": split_metrics,
        "model": str(model_path),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    result = train(args.dataset, args.run_dir)
    test = result["splits"]["test"]
    print(
        "Test MAE: persistence={p:.3f}, model={m:.3f}; rows={rows}".format(
            p=test["persistence"]["mae"], m=test["hist_gradient_boosting"]["mae"], rows=test["rows"]
        )
    )


if __name__ == "__main__":
    main()
