#!/usr/bin/env python3
"""Compare saved sequence and XGBoost models on one identical sequence cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from train_multistation_baseline import _feature_frame
    from train_sequence_model import (
        LSTMForecaster,
        SEQUENCE_FEATURES,
        TCNForecaster,
        _make_sequences,
    )
except ModuleNotFoundError:
    _BASELINE_PATH = Path(__file__).with_name("train_multistation_baseline.py")
    _BASELINE_SPEC = importlib.util.spec_from_file_location("train_multistation_baseline", _BASELINE_PATH)
    assert _BASELINE_SPEC and _BASELINE_SPEC.loader
    _BASELINE_MODULE = importlib.util.module_from_spec(_BASELINE_SPEC)
    _BASELINE_SPEC.loader.exec_module(_BASELINE_MODULE)
    _feature_frame = _BASELINE_MODULE._feature_frame

    _SEQUENCE_PATH = Path(__file__).with_name("train_sequence_model.py")
    _SEQUENCE_SPEC = importlib.util.spec_from_file_location("train_sequence_model", _SEQUENCE_PATH)
    assert _SEQUENCE_SPEC and _SEQUENCE_SPEC.loader
    _SEQUENCE_MODULE = importlib.util.module_from_spec(_SEQUENCE_SPEC)
    import sys

    sys.modules[_SEQUENCE_SPEC.name] = _SEQUENCE_MODULE
    _SEQUENCE_SPEC.loader.exec_module(_SEQUENCE_MODULE)
    LSTMForecaster = _SEQUENCE_MODULE.LSTMForecaster
    TCNForecaster = _SEQUENCE_MODULE.TCNForecaster
    SEQUENCE_FEATURES = _SEQUENCE_MODULE.SEQUENCE_FEATURES
    _make_sequences = _SEQUENCE_MODULE._make_sequences


TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-10-01", tz="UTC")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(np.unique(actual)) > 1 else 0.0,
    }


def _sequence_predictions(
    artifact_path: Path,
    bundle,
    test_mask: np.ndarray,
    valid_positions: np.ndarray,
    model_name: str,
) -> np.ndarray:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    feature_count = len(SEQUENCE_FEATURES)
    raw_values = bundle.values[:, :, :feature_count]
    medians = np.asarray(artifact["preprocessing"]["medians"], dtype=np.float32)
    means = np.asarray(artifact["preprocessing"]["means"], dtype=np.float32)
    scales = np.asarray(artifact["preprocessing"]["scales"], dtype=np.float32)
    filled = np.where(np.isnan(raw_values), medians, raw_values)
    normalized = (filled - means) / scales
    values = np.concatenate([normalized, bundle.values[:, :, feature_count:]], axis=2).astype(np.float32)
    model_class = LSTMForecaster if model_name == "lstm" else TCNForecaster
    model = model_class(values.shape[2], len(bundle.station_ids))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(values[test_mask][valid_positions]),
            torch.from_numpy(bundle.station_indices[test_mask][valid_positions]),
        ).numpy()
    target_mean = artifact["preprocessing"]["target_log_mean"][0]
    target_scale = artifact["preprocessing"]["target_log_scale"][0]
    return np.maximum(0.0, np.expm1(prediction * target_scale + target_mean))


def compare(
    dataset_path: Path,
    xgb_artifact: Path,
    sequence_artifact: Path,
    model_name: str,
    horizon: int,
    sequence_length: int,
    stride: int,
    output_path: Path,
) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    target = f"target_pm25_{horizon}h"
    frame = frame.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    bundle = _make_sequences(frame, horizon, sequence_length, stride)
    test_mask = bundle.issue_times >= np.datetime64(VALIDATION_END.to_datetime64())
    station_values = np.asarray(bundle.station_ids, dtype=object)[bundle.station_indices[test_mask]]
    issue_times = pd.to_datetime(bundle.issue_times[test_mask], utc=True)
    cohort = pd.DataFrame({"station_id": station_values, "timestamp_utc": issue_times})

    scored_frame = frame.dropna(subset=["pm25_current"]).reset_index(drop=True)
    scored_features, feature_columns, _ = _feature_frame(scored_frame)
    lookup = pd.MultiIndex.from_frame(scored_frame[["station_id", "timestamp_utc"]])
    positions = lookup.get_indexer(pd.MultiIndex.from_frame(cohort))
    valid = positions >= 0
    if not valid.any():
        raise ValueError("no sequence test rows have valid current PM2.5 for persistence comparison")
    positions_valid = positions[valid]
    actual = scored_frame.iloc[positions_valid][target].to_numpy(dtype=float)
    persistence = scored_frame.iloc[positions_valid]["pm25_current"].to_numpy(dtype=float)

    xgb = joblib.load(xgb_artifact)
    xgb_prediction = xgb["model"].predict(scored_features.iloc[positions_valid][feature_columns])
    sequence_prediction = _sequence_predictions(
        sequence_artifact,
        bundle,
        test_mask,
        np.flatnonzero(valid),
        model_name,
    )
    result: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "horizon_hours": horizon,
        "model": model_name,
        "cohort_policy": "test sequence windows with valid issue-time PM2.5, same station/timestamp keys for all models",
        "rows": int(len(actual)),
        "persistence": _metrics(actual, persistence),
        "xgboost": _metrics(actual, xgb_prediction),
        model_name: _metrics(actual, sequence_prediction),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/delhi_multistation_forecast.parquet"))
    parser.add_argument("--xgb-artifact", type=Path, default=Path("data/runs/multistation_xgboost/model_6h.joblib"))
    parser.add_argument("--sequence-artifact", type=Path, default=Path("data/runs/sequence_models/tcn_6h.pt"))
    parser.add_argument("--model", choices=("lstm", "tcn"), default="tcn")
    parser.add_argument("--horizon", type=int, choices=(1, 3, 6, 12, 24), default=6)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("data/runs/sequence_cohort_comparison_6h.json"))
    args = parser.parse_args()
    result = compare(
        args.dataset,
        args.xgb_artifact,
        args.sequence_artifact,
        args.model,
        args.horizon,
        args.sequence_length,
        args.stride,
        args.output,
    )
    for model_name in ("persistence", "xgboost", args.model):
        print(f"{model_name}: MAE={result[model_name]['mae']:.3f} RMSE={result[model_name]['rmse']:.3f}")


if __name__ == "__main__":
    main()
