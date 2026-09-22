#!/usr/bin/env python3
"""Train a compact spatial graph neural-network challenger.

Each CPCB station is a graph node. At every issue hour the model combines the
node's recent pollutant/weather state with messages from nearby stations. The
same chronological split used by the other models keeps the comparison fair.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from delhi_aircast.aqi import pm25_proxy

try:
    from train_multistation_baseline import _feature_frame
except ModuleNotFoundError:
    from scripts.train_multistation_baseline import _feature_frame


FEATURES = (
    "pm25_current",
    "pm25_lag_1h",
    "pm25_lag_3h",
    "temperature_c_current",
    "relative_humidity_pct_current",
    "wind_speed_mps_current",
    "rain_mm_current",
    "hour_sin",
    "hour_cos",
)
TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-10-01", tz="UTC")


def _graph(registry: pd.DataFrame) -> torch.Tensor:
    points = registry.set_index("station_id").loc[registry["station_id"]]
    coords = points[["latitude", "longitude"]].to_numpy(float)
    distances = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2) ** 0.5
    distances[~np.isfinite(distances)] = np.inf
    adjacency = np.zeros_like(distances)
    for i in range(len(coords)):
        nearest = np.argsort(distances[i])[: min(6, len(coords))]
        adjacency[i, nearest] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(len(coords))
    adjacency /= adjacency.sum(axis=1, keepdims=True).clip(min=1.0)
    return torch.tensor(adjacency, dtype=torch.float32)


class SpatialGNN(nn.Module):
    def __init__(self, feature_count: int, hidden: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.input = nn.Linear(feature_count, hidden)
        self.message = nn.Linear(hidden, hidden)
        self.node_embedding = nn.Parameter(torch.zeros(adjacency.shape[0], hidden))
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input(values))
        hidden = hidden + self.node_embedding
        messages = torch.relu(self.message(torch.matmul(self.adjacency, hidden)))
        hidden = torch.relu(self.norm(hidden + messages))
        return self.output(hidden).squeeze(-1)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(actual, predicted)), "rmse": float(mean_squared_error(actual, predicted) ** 0.5), "r2": float(r2_score(actual, predicted))}


def _aqi_metrics(actual_pm25: np.ndarray, predicted_pm25: np.ndarray) -> dict[str, object]:
    actual_results = [pm25_proxy(value) for value in actual_pm25]
    predicted_results = [pm25_proxy(value) for value in predicted_pm25]
    actual_aqi = np.asarray([result.value for result in actual_results], dtype=float)
    predicted_aqi = np.asarray([result.value for result in predicted_results], dtype=float)
    actual_categories = [result.category for result in actual_results]
    predicted_categories = [result.category for result in predicted_results]
    return {
        "proxy_aqi_mae": float(mean_absolute_error(actual_aqi, predicted_aqi)),
        "category_accuracy": float(accuracy_score(actual_categories, predicted_categories)),
        "category_names": sorted(set(actual_categories)),
    }


def train(dataset: Path, registry_path: Path, output: Path, horizon: int, epochs: int, lr: float, patience: int) -> dict[str, object]:
    frame = pd.read_parquet(dataset)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    station_ids = sorted(frame["station_id"].astype(str).unique())
    registry = pd.read_csv(registry_path)
    registry = registry[registry["station_id"].astype(str).isin(station_ids)].copy()
    registry = registry.dropna(subset=["latitude", "longitude"]).sort_values("station_id")
    station_ids = registry["station_id"].astype(str).tolist()
    frame = frame[frame["station_id"].astype(str).isin(station_ids)].copy()
    frame["station_id"] = frame["station_id"].astype(str)
    times = pd.DatetimeIndex(sorted(frame["timestamp_utc"].unique()))
    time_index = {value: i for i, value in enumerate(times)}
    values = np.full((len(times), len(station_ids), len(FEATURES)), np.nan, dtype=np.float32)
    targets = np.full((len(times), len(station_ids)), np.nan, dtype=np.float32)
    station_index = {sid: i for i, sid in enumerate(station_ids)}
    for row in frame.itertuples(index=False):
        t = time_index.get(row.timestamp_utc)
        s = station_index.get(row.station_id)
        if t is None or s is None:
            continue
        values[t, s] = [getattr(row, col, np.nan) for col in FEATURES]
        targets[t, s] = getattr(row, f"target_pm25_{horizon}h")
    train_mask = times < TRAIN_END
    validation_mask = (times >= TRAIN_END) & (times < VALIDATION_END)
    test_mask = times >= VALIDATION_END
    medians = np.nanmedian(values[train_mask], axis=(0, 1))
    medians = np.where(np.isfinite(medians), medians, 0.0)
    current_pm25 = values[:, :, 0].copy()
    values = np.where(np.isnan(values), medians[None, None, :], values)
    means = values[train_mask].mean(axis=(0, 1))
    scales = values[train_mask].std(axis=(0, 1)).clip(min=1e-6)
    values = (values - means[None, None, :]) / scales[None, None, :]
    valid_train = train_mask & np.isfinite(targets).any(axis=1)
    valid_validation = validation_mask & np.isfinite(targets).any(axis=1)
    valid_test = test_mask & np.isfinite(targets).any(axis=1)
    x = torch.tensor(values, dtype=torch.float32)
    log_targets = np.log1p(np.maximum(targets, 0.0))
    target_mean = float(np.nanmean(log_targets[train_mask]))
    target_scale = float(np.nanstd(log_targets[train_mask])) or 1.0
    normalized_targets = (log_targets - target_mean) / target_scale
    y = torch.tensor(np.nan_to_num(normalized_targets, nan=0.0), dtype=torch.float32)
    graph = _graph(registry)
    model = SpatialGNN(len(FEATURES), 48, graph)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_indices = np.flatnonzero(valid_train)
    validation_indices = np.flatnonzero(valid_validation)
    best_validation_mae = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        permutation = np.random.default_rng(42 + epoch).permutation(train_indices)
        for start in range(0, len(permutation), 128):
            batch = permutation[start : start + 128]
            prediction = model(x[batch])
            mask = torch.tensor(np.isfinite(targets[batch]), dtype=torch.bool)
            loss = nn.functional.smooth_l1_loss(prediction[mask], y[batch][mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_normalized = model(x[validation_indices]).numpy()
        validation_prediction = np.maximum(0.0, np.expm1(validation_normalized * target_scale + target_mean))
        validation_actual = targets[validation_indices]
        validation_valid = np.isfinite(validation_actual)
        validation_mae = float(mean_absolute_error(validation_actual[validation_valid], validation_prediction[validation_valid]))
        history.append({"epoch": epoch + 1, "validation_mae": validation_mae})
        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise ValueError("no valid validation predictions were produced")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        normalized_prediction = model(x[np.flatnonzero(valid_test)]).numpy()
    predicted = np.maximum(0.0, np.expm1(normalized_prediction * target_scale + target_mean))
    actual = targets[valid_test]
    current_test = current_pm25[valid_test]
    valid = np.isfinite(actual) & np.isfinite(current_test)
    persistence = current_test[valid]
    predicted_valid = predicted[valid]
    actual_valid = actual[valid]

    # Score the existing final XGBoost on these same station-hour keys.
    score_frame = frame.reset_index(drop=True)
    score_frame["timestamp_utc"] = pd.to_datetime(score_frame["timestamp_utc"], utc=True)
    xgb_bundle = joblib.load(Path("data/runs/final_xgboost") / f"model_{horizon}h.joblib")
    test_rows = (score_frame["timestamp_utc"] >= VALIDATION_END) & score_frame[f"target_pm25_{horizon}h"].notna() & score_frame["pm25_current"].notna()
    xgb_rows = score_frame.loc[test_rows].copy()
    xgb_features, _, _ = _feature_frame(xgb_rows)
    xgb_matrix = xgb_features.reindex(columns=xgb_bundle["feature_columns"], fill_value=0.0).astype(float)
    xgb_prediction = xgb_bundle["model"].predict(xgb_matrix)
    row_keys = pd.MultiIndex.from_arrays([times.repeat(len(station_ids)), station_ids * len(times)])
    xgb_grid = np.full((len(times), len(station_ids)), np.nan, dtype=np.float32)
    xgb_locations = row_keys.get_indexer(pd.MultiIndex.from_frame(xgb_rows[["timestamp_utc", "station_id"]]))
    xgb_valid_rows = xgb_locations >= 0
    xgb_grid.ravel()[xgb_locations[xgb_valid_rows]] = xgb_prediction[xgb_valid_rows]
    xgb_test = xgb_grid[np.flatnonzero(valid_test)][valid]

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "spatial_gnn",
        "horizon_hours": horizon,
        "feature_columns": list(FEATURES),
        "station_ids": station_ids,
        "test_rows": int(valid.sum()),
        "metrics": _metrics(actual_valid, predicted_valid),
        "aqi_proxy": _aqi_metrics(actual_valid, predicted_valid),
        "persistence": _metrics(actual_valid, persistence),
        "persistence_aqi_proxy": _aqi_metrics(actual_valid, persistence),
        "xgboost_same_cohort": _metrics(actual_valid, xgb_test),
        "xgboost_aqi_proxy_same_cohort": _aqi_metrics(actual_valid, xgb_test),
        "validation_best_mae": best_validation_mae,
        "training_history": history,
        "graph": "undirected six-nearest-station adjacency with self-loops",
        "target_transform": "standardized log1p fitted on pre-2025-07-01 targets",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_columns": list(FEATURES), "station_ids": station_ids, "means": means.tolist(), "scales": scales.tolist(), "target_mean": target_mean, "target_scale": target_scale, "adjacency": graph.tolist(), "horizon_hours": horizon}, output.with_suffix(".pt"))
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/delhi_unified_forecast.parquet"))
    parser.add_argument("--registry", type=Path, default=Path("data/processed/cpcb_panel_quality/station_registry.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/runs/spatial_gnn/model_6h"))
    parser.add_argument("--horizon", type=int, choices=(1, 3, 6, 12, 24), default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    result = train(args.dataset, args.registry, args.output, args.horizon, args.epochs, args.lr, args.patience)
    print(json.dumps({key: result[key] for key in ("metrics", "persistence", "xgboost_same_cohort", "test_rows")}, indent=2))


if __name__ == "__main__":
    main()
