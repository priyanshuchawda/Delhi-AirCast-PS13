#!/usr/bin/env python3
"""Run rolling-origin, station-aware PM2.5 backtests."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from train_multistation_baseline import _feature_frame
except ModuleNotFoundError:
    import importlib.util

    _BASELINE_PATH = Path(__file__).with_name("train_multistation_baseline.py")
    _BASELINE_SPEC = importlib.util.spec_from_file_location("train_multistation_baseline", _BASELINE_PATH)
    assert _BASELINE_SPEC and _BASELINE_SPEC.loader
    _BASELINE_MODULE = importlib.util.module_from_spec(_BASELINE_SPEC)
    _BASELINE_SPEC.loader.exec_module(_BASELINE_MODULE)
    _feature_frame = _BASELINE_MODULE._feature_frame


DEFAULT_DATASET = Path("data/processed/delhi_multistation_forecast.parquet")
DEFAULT_OUTPUT = Path("data/runs/rolling_backtests.json")
ORIGINS = (
    pd.Timestamp("2024-10-01", tz="UTC"),
    pd.Timestamp("2025-01-01", tz="UTC"),
    pd.Timestamp("2025-04-01", tz="UTC"),
    pd.Timestamp("2025-07-01", tz="UTC"),
    pd.Timestamp("2025-10-01", tz="UTC"),
)


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if actual.nunique() > 1 else 0.0,
    }


def _folds(origins: tuple[pd.Timestamp, ...] = ORIGINS) -> list[dict[str, str]]:
    return [
        {
            "fold": f"origin_{origin.strftime('%Y%m%d')}",
            "train_end_utc": origin.isoformat(),
            "test_end_utc": (origin + pd.DateOffset(months=3)).isoformat(),
        }
        for origin in origins
    ]


def _xgboost_model(n_estimators: int, n_jobs: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=42,
    )


def run_backtest(
    dataset_path: Path,
    horizon: int,
    model_name: str = "persistence",
    output_path: Path = DEFAULT_OUTPUT,
    n_estimators: int = 150,
    n_jobs: int = 6,
) -> dict[str, object]:
    target = f"target_pm25_{horizon}h"
    frame = pd.read_parquet(dataset_path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.dropna(subset=[target, "pm25_current"]).sort_values("timestamp_utc").reset_index(drop=True)
    features, feature_columns, _ = _feature_frame(frame)
    features = features.astype(float)
    fold_results: list[dict[str, object]] = []

    for fold in _folds():
        train_end = pd.Timestamp(fold["train_end_utc"])
        test_end = pd.Timestamp(fold["test_end_utc"])
        train_mask = frame["timestamp_utc"] < train_end
        test_mask = (frame["timestamp_utc"] >= train_end) & (frame["timestamp_utc"] < test_end)
        if not train_mask.any() or not test_mask.any():
            continue
        actual = frame.loc[test_mask, target]
        persistence = frame.loc[test_mask, "pm25_current"]
        result: dict[str, object] = {
            **fold,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "persistence": _metrics(actual, persistence),
        }
        if model_name == "xgboost":
            model = _xgboost_model(n_estimators, n_jobs)
            model.fit(features.loc[train_mask, feature_columns], frame.loc[train_mask, target])
            predicted = pd.Series(model.predict(features.loc[test_mask, feature_columns]), index=actual.index)
            result["xgboost"] = _metrics(actual, predicted)
            result["station_metrics"] = {
                str(station_id): {
                    "rows": int(len(frame.loc[test_mask & frame["station_id"].eq(station_id)])),
                    "persistence_mae": float(
                        mean_absolute_error(
                            frame.loc[test_mask & frame["station_id"].eq(station_id), target],
                            frame.loc[test_mask & frame["station_id"].eq(station_id), "pm25_current"],
                        )
                    ),
                    "xgboost_mae": float(
                        mean_absolute_error(
                            frame.loc[test_mask & frame["station_id"].eq(station_id), target],
                            predicted.loc[test_mask & frame["station_id"].eq(station_id)],
                        )
                    ),
                }
                for station_id in sorted(frame.loc[test_mask, "station_id"].unique())
            }
        fold_results.append(result)

    summary: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "horizon_hours": horizon,
        "model": model_name,
        "fold_policy": "expanding train window with quarterly three-month test windows",
        "folds": fold_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--horizon", type=int, choices=(1, 3, 6, 12, 24), default=1)
    parser.add_argument("--model", choices=("persistence", "xgboost"), default="persistence")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()
    summary = run_backtest(
        args.dataset,
        args.horizon,
        args.model,
        args.output,
        args.n_estimators,
        args.n_jobs,
    )
    for fold in summary["folds"]:
        metrics = fold[args.model]
        print(f"{fold['fold']}: rows={fold['test_rows']} MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f}")


if __name__ == "__main__":
    main()
