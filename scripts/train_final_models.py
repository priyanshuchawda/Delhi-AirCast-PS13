#!/usr/bin/env python3
"""Train and evaluate the final unified-feature PM2.5 models.

Final artifacts are fit on the full pre-test history (train + validation,
everything before 2025-10-01) so served models use every leakage-checked
observation available before the test tail. The untouched test tail
(>= 2025-10-01) is used only for the final per-horizon evaluation report.

Dataset: data/processed/delhi_unified_forecast.parquet (primary CPCB/OpenCity
panel + as-of Open-Meteo weather and CAMS PM2.5/PM10 with the 6h publish lag +
prior-day NASA FIRMS fire features).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

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


DEFAULT_DATASET = Path("data/processed/delhi_unified_forecast.parquet")
DEFAULT_RUN_DIR = Path("data/runs/final_xgboost")
HORIZONS = (1, 3, 6, 12, 24)
VALIDATION_END = pd.Timestamp("2025-10-01", tz="UTC")
FEATURE_SET = "unified-spatial-temporal"
POINT_IN_TIME_POLICY = (
    "station and lag/rolling pollutant features are stamped at the issue hour t; "
    "central weather/CAMS are as-of aligned to the newest reading with a stamp "
    "not later than t; CAMS concentrations additionally carry a 6 h published "
    "lag; FIRMS fire features are previous-day only."
)


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if actual.nunique() > 1 else 0.0,
    }


def _model(n_jobs: int):
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=42,
    )


def train_one(dataset_path: Path, run_dir: Path, horizon: int, n_jobs: int = 2) -> dict[str, object]:
    target = f"target_pm25_{horizon}h"
    frame = pd.read_parquet(dataset_path).sort_values("timestamp_utc").reset_index(drop=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.dropna(subset=[target, "pm25_current"]).reset_index(drop=True)
    feature_frame, feature_columns, station_codes = _feature_frame(frame)
    feature_frame = feature_frame.astype(float)

    fit_mask = frame["timestamp_utc"] < VALIDATION_END
    test_mask = frame["timestamp_utc"] >= VALIDATION_END
    if not fit_mask.any() or not test_mask.any():
        raise ValueError("final fit/test split produced an empty partition")

    model = _model(n_jobs)
    model.fit(feature_frame.loc[fit_mask, feature_columns], frame.loc[fit_mask, target])

    actual = frame.loc[test_mask, target]
    predicted = pd.Series(model.predict(feature_frame.loc[test_mask, feature_columns]), index=actual.index)
    persistence = frame.loc[test_mask, "pm25_current"]

    station_metrics: dict[str, dict[str, float]] = {}
    for station_id, station_mask in frame.loc[test_mask].groupby("station_id").groups.items():
        idx = station_mask.intersection(test_mask.index)
        if len(idx) == 0:
            continue
        station_actual = actual.loc[idx]
        station_predicted = predicted.loc[idx]
        station_persistence = persistence.loc[idx]
        station_metrics[str(station_id)] = {
            "rows": int(len(idx)),
            "persistence_mae": float(mean_absolute_error(station_actual, station_persistence)),
            "xgboost_mae": float(mean_absolute_error(station_actual, station_predicted)),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"model_{horizon}h.joblib"
    artifact: dict[str, object] = {
        "model": model,
        "feature_columns": feature_columns,
        "station_codes": station_codes,
        "horizon_hours": horizon,
        "feature_set": FEATURE_SET,
        "point_in_time_policy": POINT_IN_TIME_POLICY,
        "fit_interval": "train+validation < 2025-10-01; test tail reserved for evaluation",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    joblib.dump(artifact, model_path)

    summary: dict[str, object] = {
        "horizon_hours": horizon,
        "generated_at": artifact["generated_at"],
        "dataset": str(dataset_path),
        "model": str(model_path),
        "feature_set": FEATURE_SET,
        "fit_rows": int(fit_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_interval": (
            frame.loc[test_mask, "timestamp_utc"].min().isoformat(),
            frame.loc[test_mask, "timestamp_utc"].max().isoformat(),
        ),
        "persistence": _metrics(actual, persistence),
        "xgboost": _metrics(actual, predicted),
        "station_metrics": station_metrics,
    }
    (run_dir / f"result_{horizon}h.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()

    results = {h: train_one(args.dataset, args.run_dir, h, args.n_jobs) for h in args.horizons}
    summary_path = args.run_dir / "final_evaluation.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {"generated_at": datetime.now(UTC).isoformat(), "feature_set": FEATURE_SET, "horizons": results},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")
    for h, result in results.items():
        test = result["xgboost"]
        persistence = result["persistence"]
        gain = (persistence["mae"] - test["mae"]) / persistence["mae"] * 100
        print(
            f"{h}h  test MAE persistence={persistence['mae']:.3f} xgboost={test['mae']:.3f} "
            f"RMSE={test['rmse']:.3f} R2={test['r2']:.4f} vs-persistence {gain:+.1f}% rows={result['test_rows']}"
        )


if __name__ == "__main__":
    main()