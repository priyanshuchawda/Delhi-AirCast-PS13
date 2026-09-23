#!/usr/bin/env python3
"""Train pollutant forecasts and evaluate CPCB AQI composition on a time holdout.

This first implementation focuses on one configurable direct horizon. CPCB
pollutant concentrations remain the labels; the AQI engine combines available
forecast sub-indices only when the minimum input requirements are satisfied.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

try:
    from train_multistation_baseline import _feature_frame
except ModuleNotFoundError:
    try:
        from scripts.train_multistation_baseline import _feature_frame
    except ModuleNotFoundError:
        import importlib.util

        _BASELINE_PATH = Path(__file__).with_name("train_multistation_baseline.py")
        _BASELINE_SPEC = importlib.util.spec_from_file_location("train_multistation_baseline", _BASELINE_PATH)
        assert _BASELINE_SPEC and _BASELINE_SPEC.loader
        _BASELINE_MODULE = importlib.util.module_from_spec(_BASELINE_SPEC)
        _BASELINE_SPEC.loader.exec_module(_BASELINE_MODULE)
        _feature_frame = _BASELINE_MODULE._feature_frame

from delhi_aircast.aqi import calculate_aqi, pm25_proxy


DEFAULT_DATASET = Path("data/processed/delhi_unified_forecast.parquet")
DEFAULT_RUN_DIR = Path("data/runs/multipollutant_aqi_diagnostic")
POLLUTANTS = ("pm25", "pm10", "no2", "co", "o3")
TEST_START = pd.Timestamp("2025-10-01", tz="UTC")
TRAIN_STRIDE = 3


def _model(n_jobs: int, full_fit: bool = False) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror", n_estimators=350 if full_fit else 140, learning_rate=0.05,
        max_depth=8 if full_fit else 7, min_child_weight=8, subsample=0.9, colsample_bytree=0.85,
        reg_lambda=2.0, tree_method="hist", n_jobs=n_jobs, random_state=42,
    )


def _training_mask(timestamps: pd.Series, horizon: int) -> pd.Series:
    """Return a purged pre-test issue-time mask for direct-horizon labels."""
    return timestamps + pd.Timedelta(hours=horizon) < TEST_START


def _aqi_metrics(actual: list[dict[str, float]], predicted: list[dict[str, float]]) -> dict[str, float | int]:
    actual_results = [calculate_aqi(row) for row in actual]
    predicted_results = [calculate_aqi(row) for row in predicted]
    valid = [i for i, result in enumerate(actual_results) if result.value is not None and predicted_results[i].value is not None]
    if not valid:
        return {"rows": 0, "aqi_mae": float("nan"), "category_accuracy": float("nan")}
    return {
        "rows": len(valid),
        "aqi_mae": float(mean_absolute_error([actual_results[i].value for i in valid], [predicted_results[i].value for i in valid])),
        "category_accuracy": float(accuracy_score([actual_results[i].category for i in valid], [predicted_results[i].category for i in valid])),
    }


def _proxy_against_composite(actual: list[dict[str, float]], predicted: list[dict[str, float]]) -> dict[str, float | int]:
    pairs = [(a, p) for a, p in zip(actual, predicted) if "pm25" in a and "pm25" in p]
    actual_results = [calculate_aqi(row) for row, _ in pairs]
    proxy_results = [pm25_proxy(pred["pm25"]) for _, pred in pairs]
    valid = [i for i, result in enumerate(actual_results) if result.value is not None and proxy_results[i].value is not None]
    if not valid:
        return {"rows": 0, "aqi_mae": float("nan"), "category_accuracy": float("nan")}
    return {
        "rows": len(valid),
        "aqi_mae": float(mean_absolute_error([actual_results[i].value for i in valid], [proxy_results[i].value for i in valid])),
        "category_accuracy": float(accuracy_score([actual_results[i].category for i in valid], [proxy_results[i].category for i in valid])),
    }


def train(dataset: Path, run_dir: Path, horizon: int, n_jobs: int = 4, reuse_models: bool = False, train_stride: int = TRAIN_STRIDE) -> dict[str, object]:
    frame = pd.read_parquet(dataset).sort_values("timestamp_utc").reset_index(drop=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    features, columns, station_codes = _feature_frame(frame)
    features = features.astype(float)
    # Purge the horizon at the boundary so no training label lands in the
    # held-out test interval (purged chronological split).
    train_rows = _training_mask(frame.timestamp_utc, horizon)
    station_row = frame.groupby(frame.station_id.astype(str), sort=False).cumcount()
    if train_stride < 1:
        raise ValueError("train_stride must be >= 1")
    train_rows &= station_row.mod(train_stride).eq(0)
    test_rows = frame.timestamp_utc >= TEST_START
    if not train_rows.any() or not test_rows.any():
        raise ValueError("pre-test or test partition is empty")

    predictions: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    persistence: dict[str, np.ndarray] = {}
    model_metrics: dict[str, object] = {}
    run_dir.mkdir(parents=True, exist_ok=True)

    for pollutant in POLLUTANTS:
        current_column = f"{pollutant}_current"
        target_column = f"target_{pollutant}_{horizon}h"
        if current_column not in frame:
            continue
        # Direct future pollutant label, respecting station boundaries.
        target = frame.groupby(frame.station_id.astype(str), sort=False)[current_column].shift(-horizon)
        usable_train = train_rows & target.notna() & frame[current_column].notna()
        usable_test = test_rows & target.notna() & frame[current_column].notna()
        if not usable_train.any() or not usable_test.any():
            continue

        artifact_path = run_dir / f"{pollutant}_{horizon}h.joblib"
        if pollutant == "pm25":
            saved = joblib.load(Path("data/runs/final_xgboost") / f"model_{horizon}h.joblib")
            model = saved["model"]
            model_columns = saved["feature_columns"]
        elif reuse_models:
            saved = joblib.load(artifact_path)
            model = saved["model"]
            model_columns = saved["feature_columns"]
        else:
            model_columns = columns
            model = _model(n_jobs, full_fit=train_stride == 1)
            model.fit(features.loc[usable_train, model_columns], target.loc[usable_train])
            joblib.dump({"model": model, "feature_columns": model_columns, "station_codes": station_codes,
                         "horizon_hours": horizon, "pollutant": pollutant}, artifact_path)

        test_prediction = np.maximum(0.0, model.predict(features.loc[usable_test, model_columns]))
        full_prediction = np.full(len(frame), np.nan, dtype=float)
        full_prediction[np.flatnonzero(usable_test.to_numpy())] = test_prediction
        predictions[pollutant] = full_prediction
        targets[pollutant] = target.to_numpy(dtype=float)
        persistence[pollutant] = frame[current_column].to_numpy(dtype=float)
        actual = target.loc[usable_test].to_numpy(dtype=float)
        model_metrics[pollutant] = {
            "train_rows": int(usable_train.sum()),
            "test_rows": int(usable_test.sum()),
            "test_mae": float(mean_absolute_error(actual, test_prediction)),
            "test_rmse": float(mean_squared_error(actual, test_prediction) ** 0.5),
            "model": "existing final XGBoost artifact" if pollutant == "pm25" else str(artifact_path),
        }

    # Score AQI on a matched cohort: for each row use only pollutants with an
    # observed future label and a comparable issue-time persistence value.
    actual_rows: list[dict[str, float]] = []
    forecast_rows: list[dict[str, float]] = []
    persistence_rows: list[dict[str, float]] = []
    test_positions = np.flatnonzero(test_rows.to_numpy())
    for index in test_positions:
        available = [p for p in predictions if np.isfinite(targets[p][index]) and np.isfinite(predictions[p][index]) and np.isfinite(persistence[p][index])]
        current_row = {p: float(persistence[p][index]) for p in available}
        if len(available) < 3 or not ({"pm25", "pm10"} & set(available)):
            continue
        actual_rows.append({p: float(targets[p][index]) for p in available})
        forecast_rows.append({p: float(predictions[p][index]) for p in available})
        persistence_rows.append(current_row)

    proxy_pairs = [(actual, predicted, persistent) for actual, predicted, persistent in zip(actual_rows, forecast_rows, persistence_rows) if "pm25" in actual]
    actual_pm25_cohort = [actual for actual, _, _ in proxy_pairs]
    predicted_pm25_cohort = [predicted for _, predicted, _ in proxy_pairs]
    persistence_pm25_cohort = [persistent for _, _, persistent in proxy_pairs]
    proxy_aqi_actual = [pm25_proxy(actual["pm25"]) for actual, _, _ in proxy_pairs]
    proxy_aqi_predicted = [pm25_proxy(predicted["pm25"]) for _, predicted, _ in proxy_pairs]

    result: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset),
        "horizon_hours": horizon,
        "split": {"fit_before_utc": TEST_START.isoformat(), "test_from_utc": TEST_START.isoformat()},
        "diagnostic_sampling": "Full eligible pre-test history." if train_stride == 1 else f"Every {train_stride}th eligible training row per station; validation/test observations are not thinned.",
        "pollutant_models": model_metrics,
        "aqi_forecast": _aqi_metrics(actual_rows, forecast_rows),
        "aqi_persistence": _aqi_metrics(actual_rows, persistence_rows),
        "aqi_forecast_pm25_cohort": _aqi_metrics(actual_pm25_cohort, predicted_pm25_cohort),
        "aqi_persistence_pm25_cohort": _aqi_metrics(actual_pm25_cohort, persistence_pm25_cohort),
        "pm25_proxy_vs_composite_actual": _proxy_against_composite(actual_rows, forecast_rows),
        "pm25_persistence_proxy_vs_composite_actual": _proxy_against_composite(actual_rows, persistence_rows),
        "aqi_input_pollutants": list(POLLUTANTS),
        "aqi_policy": "At each row, compute from the matched observed-future/forecast/persistence pollutant subset; require at least 3 and PM2.5 or PM10.",
        "aqi_note": "Official CPCB calculation over the available forecast pollutant subset; omitted SO2/NH3 are not modelled in this first experiment. It is not a PM2.5-only proxy.",
        "matched_aqi_rows": int(len(actual_rows)),
        "pm25_proxy_comparison": {
            "rows": len(proxy_pairs),
            "aqi_mae": float(mean_absolute_error([r.value for r in proxy_aqi_actual], [r.value for r in proxy_aqi_predicted])) if proxy_pairs else None,
            "category_accuracy": float(accuracy_score([r.category for r in proxy_aqi_actual], [r.category for r in proxy_aqi_predicted])) if proxy_pairs else None,
        },
    }
    (run_dir / f"evaluation_{horizon}h.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--horizon", type=int, choices=(1, 3, 6, 12, 24), default=6)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--reuse-models", action="store_true", help="Score existing diagnostic artifacts without retraining")
    parser.add_argument("--train-stride", type=int, default=TRAIN_STRIDE, help="Keep every nth training row per station; use 1 for all rows")
    args = parser.parse_args()
    result = train(args.dataset, args.run_dir, args.horizon, args.n_jobs, args.reuse_models, args.train_stride)
    print(json.dumps({"horizon": result["horizon_hours"], "pollutants": result["pollutant_models"],
                      "composite_aqi": result["aqi_forecast"], "persistence_aqi": result["aqi_persistence"],
                      "composite_aqi_pm25_cohort": result["aqi_forecast_pm25_cohort"],
                      "pm25_proxy_vs_composite": result["pm25_proxy_vs_composite_actual"]}, indent=2))


if __name__ == "__main__":
    main()
