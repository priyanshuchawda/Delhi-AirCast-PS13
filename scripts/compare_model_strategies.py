#!/usr/bin/env python3
"""Compare feature families, boosted-tree models, and validation-tuned blends.

All candidates share chronological train/validation/test windows. The held-out
test tail is scored only after the validation interval selects the blend.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

try:
    from train_multistation_baseline import _feature_frame
except ModuleNotFoundError:
    try:
        from scripts.train_multistation_baseline import _feature_frame
    except ModuleNotFoundError:
        _BASELINE_PATH = Path(__file__).with_name("train_multistation_baseline.py")
        _BASELINE_SPEC = importlib.util.spec_from_file_location("train_multistation_baseline", _BASELINE_PATH)
        assert _BASELINE_SPEC and _BASELINE_SPEC.loader
        _BASELINE_MODULE = importlib.util.module_from_spec(_BASELINE_SPEC)
        _BASELINE_SPEC.loader.exec_module(_BASELINE_MODULE)
        _feature_frame = _BASELINE_MODULE._feature_frame

from delhi_aircast.aqi import pm25_proxy


DEFAULT_DATASET = Path("data/processed/delhi_unified_forecast.parquet")
DEFAULT_OUTPUT = Path("data/runs/model_strategy_comparison.json")
TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-10-01", tz="UTC")
HORIZONS = (1, 6, 24)
TRAIN_STRIDE = 3


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_aqi = [pm25_proxy(value).value for value in actual]
    predicted_aqi = [pm25_proxy(value).value for value in predicted]
    actual_categories = [pm25_proxy(value).category for value in actual]
    predicted_categories = [pm25_proxy(value).category for value in predicted]
    return {
        "pm25_mae": float(mean_absolute_error(actual, predicted)),
        "pm25_rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "aqi_proxy_mae": float(mean_absolute_error(actual_aqi, predicted_aqi)),
        "aqi_proxy_category_accuracy": float(accuracy_score(actual_categories, predicted_categories)),
    }


def _feature_sets(columns: list[str]) -> dict[str, list[str]]:
    columns = [c for c in columns if not c.startswith("target_")]
    families = {
        "weather": [c for c in columns if c.startswith("wx_")],
        "cams": [c for c in columns if c.startswith("cams_")],
        "firms": [c for c in columns if c.startswith("firms_")],
    }
    core = [c for c in columns if not any(c.startswith(prefix) for prefix in ("wx_", "cams_", "firms_"))]
    return {
        "CPCB_sensor_spatiotemporal": core,
        "core_plus_weather": core + families["weather"],
        "core_plus_weather_cams": core + families["weather"] + families["cams"],
        "all_sources": columns,
    }


def _xgb(n_jobs: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror", n_estimators=160, learning_rate=0.05,
        max_depth=8, min_child_weight=8, subsample=0.9, colsample_bytree=0.85,
        reg_lambda=2.0, tree_method="hist", n_jobs=n_jobs, random_state=42,
    )


def compare(dataset: Path, output: Path, horizons: tuple[int, ...] = HORIZONS, n_jobs: int = 4) -> dict[str, object]:
    frame = pd.read_parquet(dataset).sort_values("timestamp_utc").reset_index(drop=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    features, columns, _ = _feature_frame(frame)
    features = features.astype(float)
    sets = _feature_sets(columns)
    train_mask = frame.timestamp_utc < TRAIN_END
    # Deterministic within-station thinning keeps the diagnostic affordable
    # while retaining every part of the training period and each station.
    row_number = frame.groupby(frame.station_id.astype(str), sort=False).cumcount()
    train_mask &= row_number.mod(TRAIN_STRIDE).eq(0)
    validation_mask = (frame.timestamp_utc >= TRAIN_END) & (frame.timestamp_utc < VALIDATION_END)
    test_mask = frame.timestamp_utc >= VALIDATION_END
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("chronological split has an empty train, validation, or test partition")

    reports: dict[str, object] = {}
    for horizon in horizons:
        target_column = f"target_pm25_{horizon}h"
        usable = frame[target_column].notna() & frame.pm25_current.notna()
        masks = [mask & usable for mask in (train_mask, validation_mask, test_mask)]
        actual = frame[target_column].to_numpy(dtype=float)
        persistence = frame.pm25_current.to_numpy(dtype=float)
        results: dict[str, object] = {"rows": {"train": int(masks[0].sum()), "validation": int(masks[1].sum()), "test": int(masks[2].sum())}, "feature_ablations": {}}

        for set_name, selected in sets.items():
            model = _xgb(n_jobs)
            model.fit(features.loc[masks[0], selected], actual[masks[0]])
            validation_pred = np.maximum(0.0, model.predict(features.loc[masks[1], selected]))
            test_pred = np.maximum(0.0, model.predict(features.loc[masks[2], selected]))
            results["feature_ablations"][set_name] = {
                "feature_count": len(selected),
                "validation": _score(actual[masks[1]], validation_pred),
                "test": _score(actual[masks[2]], test_pred),
            }

        selected = sets["all_sources"]
        xgb = _xgb(n_jobs)
        xgb.fit(features.loc[masks[0], selected], actual[masks[0]])
        val_xgb = np.maximum(0.0, xgb.predict(features.loc[masks[1], selected]))
        test_xgb = np.maximum(0.0, xgb.predict(features.loc[masks[2], selected]))

        hist = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.08, max_iter=50,
            max_leaf_nodes=31, l2_regularization=2.0, early_stopping=False,
            random_state=42,
        )
        hist.fit(features.loc[masks[0], selected], actual[masks[0]])
        val_hist = np.maximum(0.0, hist.predict(features.loc[masks[1], selected]))
        test_hist = np.maximum(0.0, hist.predict(features.loc[masks[2], selected]))

        weights = np.linspace(0.0, 1.0, 21)
        weight = min(weights, key=lambda w: mean_absolute_error(actual[masks[1]], w * val_xgb + (1 - w) * val_hist))
        blended_test = np.maximum(0.0, weight * test_xgb + (1 - weight) * test_hist)
        results["model_comparison"] = {
            "persistence_test": _score(actual[masks[2]], persistence[masks[2]]),
            "xgboost": {"validation": _score(actual[masks[1]], val_xgb), "test": _score(actual[masks[2]], test_xgb)},
            "hist_gradient_boosting": {"validation": _score(actual[masks[1]], val_hist), "test": _score(actual[masks[2]], test_hist)},
            "validation_tuned_blend": {
                "xgboost_weight": float(weight), "hist_gradient_boosting_weight": float(1 - weight),
                "validation": _score(actual[masks[1]], weight * val_xgb + (1 - weight) * val_hist),
                "test": _score(actual[masks[2]], blended_test),
                "selection_rule": "minimum validation PM2.5 MAE over weights 0.00..1.00 in 0.05 increments",
            },
        }
        reports[str(horizon)] = results

    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset),
        "split": {"train_before": TRAIN_END.isoformat(), "validation": [TRAIN_END.isoformat(), VALIDATION_END.isoformat()], "test_from": VALIDATION_END.isoformat()},
        "feature_groups": {"weather": "wx_*", "CAMS": "cams_*", "fires": "firms_*", "core": "CPCB pollutant/time/neighbour-derived columns"},
        "results": reports,
        "label_note": "CPCB PM2.5 is the supervised target. AQI outcomes are PM2.5-only proxies, not composite AQI.",
        "diagnostic_sampling": f"Every {TRAIN_STRIDE}rd row per station in the training interval; validation and test rows are not thinned.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=int, nargs="+", choices=(1, 3, 6, 12, 24), default=list(HORIZONS))
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    report = compare(args.dataset, args.output, tuple(args.horizons), args.n_jobs)
    for horizon, result in report["results"].items():
        print(f"{horizon}h feature ablation:")
        for name, score in result["feature_ablations"].items():
            print(f"  {name}: validation MAE={score['validation']['pm25_mae']:.3f}; test MAE={score['test']['pm25_mae']:.3f}")
        for name in ("persistence_test", "xgboost", "hist_gradient_boosting", "validation_tuned_blend"):
            score = result["model_comparison"][name]
            score = score.get("test", score)
            print(f"  {name}: test MAE={score['pm25_mae']:.3f}, AQI-proxy category accuracy={score['aqi_proxy_category_accuracy']:.3%}")


if __name__ == "__main__":
    main()
