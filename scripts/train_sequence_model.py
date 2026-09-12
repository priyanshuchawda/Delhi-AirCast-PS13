#!/usr/bin/env python3
"""Train a CPU-safe global LSTM or TCN PM2.5 sequence challenger."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATASET = Path("data/processed/delhi_multistation_forecast.parquet")
DEFAULT_RUN_DIR = Path("data/runs/sequence_models")
TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-10-01", tz="UTC")
SEQUENCE_FEATURES = (
    "pm25_current",
    "neighbor_pm25_mean_current",
    "neighbor_pm25_available_count",
    "temperature_c_current",
    "relative_humidity_pct_current",
    "wind_speed_mps_current",
    "wind_direction_deg_current",
    "rain_mm_current",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)


@dataclass(frozen=True)
class SequenceBundle:
    values: np.ndarray
    station_indices: np.ndarray
    targets: np.ndarray
    issue_times: np.ndarray
    station_ids: list[str]


def _make_sequences(frame: pd.DataFrame, horizon: int, sequence_length: int, stride: int) -> SequenceBundle:
    frame = frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    station_ids = sorted(frame["station_id"].astype(str).unique())
    station_index = {station_id: index for index, station_id in enumerate(station_ids)}
    values: list[np.ndarray] = []
    station_indices: list[int] = []
    targets: list[float] = []
    issue_times: list[np.datetime64] = []
    target_column = f"target_pm25_{horizon}h"

    for station_id, group in frame.groupby("station_id", sort=True):
        group = group.reset_index(drop=True)
        timestamps = group["timestamp_utc"].to_numpy(dtype="datetime64[ns]")
        raw_values = group[list(SEQUENCE_FEATURES)].to_numpy(dtype=np.float32)
        missing = np.isnan(raw_values).astype(np.float32)
        sequence_values = np.concatenate([raw_values, missing], axis=1)
        target_values = pd.to_numeric(group[target_column], errors="coerce").to_numpy(dtype=np.float32)
        for end in range(sequence_length - 1, len(group) - horizon, stride):
            start = end - sequence_length + 1
            target_index = end + horizon
            required = np.diff(timestamps[start : target_index + 1])
            if len(required) and not np.all(required == np.timedelta64(1, "h")):
                continue
            if not np.isfinite(target_values[end]):
                continue
            values.append(sequence_values[start : end + 1])
            station_indices.append(station_index[str(station_id)])
            targets.append(float(target_values[end]))
            issue_times.append(timestamps[end])
    if not values:
        raise ValueError("no valid contiguous sequences were created")
    return SequenceBundle(
        values=np.stack(values).astype(np.float32),
        station_indices=np.asarray(station_indices, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.float32),
        issue_times=np.asarray(issue_times, dtype="datetime64[ns]"),
        station_ids=station_ids,
    )


def _fit_preprocessing(bundle: SequenceBundle, train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    feature_count = len(SEQUENCE_FEATURES)
    train_values = bundle.values[train_mask, :, :feature_count]
    medians = np.nanmedian(train_values, axis=(0, 1))
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    filled = np.where(np.isnan(bundle.values[:, :, :feature_count]), medians, bundle.values[:, :, :feature_count])
    means = filled[train_mask].mean(axis=(0, 1))
    scales = filled[train_mask].std(axis=(0, 1))
    scales = np.where(scales > 1e-6, scales, 1.0).astype(np.float32)
    normalized = filled.copy()
    normalized = (normalized - means) / scales
    normalized = np.concatenate([normalized, bundle.values[:, :, feature_count:]], axis=2).astype(np.float32)
    log_targets = np.log1p(np.maximum(bundle.targets, 0.0))
    target_mean = float(log_targets[train_mask].mean())
    target_scale = float(log_targets[train_mask].std()) or 1.0
    preprocessing = {
        "feature_columns": list(SEQUENCE_FEATURES),
        "medians": medians.tolist(),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "target_log_mean": [target_mean],
        "target_log_scale": [target_scale],
    }
    normalized_targets = ((log_targets - target_mean) / target_scale).astype(np.float32)
    return normalized, {**preprocessing, "normalized_targets": normalized_targets.tolist()}


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, station_count: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, 8)
        self.recurrent = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size + 8), nn.Linear(hidden_size + 8, 1))

    def forward(self, values: torch.Tensor, station_indices: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(values)
        embedding = self.station_embedding(station_indices)
        return self.head(torch.cat([output[:, -1, :], embedding], dim=1)).squeeze(1)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.padding = padding

    def _trim(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :, :-self.padding] if self.padding else value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self._trim(self.activation(self.norm1(self.conv1(value))))
        value = self._trim(self.activation(self.norm2(self.conv2(value))))
        return self.activation(value + residual)


class TCNForecaster(nn.Module):
    def __init__(self, input_size: int, station_count: int, channels: int = 64) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, 8)
        self.input_projection = nn.Conv1d(input_size, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(channels, 1),
            TCNBlock(channels, 2),
            TCNBlock(channels, 4),
            TCNBlock(channels, 8),
        )
        self.head = nn.Sequential(nn.LayerNorm(channels + 8), nn.Linear(channels + 8, 1))

    def forward(self, values: torch.Tensor, station_indices: torch.Tensor) -> torch.Tensor:
        value = self.blocks(self.input_projection(values.transpose(1, 2)))
        embedding = self.station_embedding(station_indices)
        return self.head(torch.cat([value[:, :, -1], embedding], dim=1)).squeeze(1)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(np.unique(actual)) > 1 else 0.0,
    }


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    loss_function = nn.SmoothL1Loss()
    for values, stations, targets in loader:
        values, stations, targets = values.to(device), stations.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        predictions = model(values, stations)
        loss = loss_function(predictions, targets)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train(
    dataset_path: Path,
    run_dir: Path,
    model_name: str,
    horizon: int,
    sequence_length: int,
    stride: int,
    epochs: int,
    batch_size: int,
) -> dict[str, object]:
    torch.set_num_threads(min(10, os.cpu_count() or 1))
    torch.manual_seed(42)
    frame = pd.read_parquet(dataset_path)
    bundle = _make_sequences(frame, horizon, sequence_length, stride)
    train_mask = bundle.issue_times < np.datetime64(TRAIN_END.to_datetime64())
    validation_mask = (bundle.issue_times >= np.datetime64(TRAIN_END.to_datetime64())) & (
        bundle.issue_times < np.datetime64(VALIDATION_END.to_datetime64())
    )
    test_mask = bundle.issue_times >= np.datetime64(VALIDATION_END.to_datetime64())
    if min(int(train_mask.sum()), int(validation_mask.sum()), int(test_mask.sum())) == 0:
        raise ValueError("sequence time split produced an empty partition")

    normalized_values, preprocessing = _fit_preprocessing(bundle, train_mask)
    normalized_targets = np.asarray(preprocessing.pop("normalized_targets"), dtype=np.float32)
    datasets = {
        "train": TensorDataset(
            torch.from_numpy(normalized_values[train_mask]),
            torch.from_numpy(bundle.station_indices[train_mask]),
            torch.from_numpy(normalized_targets[train_mask]),
        ),
        "validation": TensorDataset(
            torch.from_numpy(normalized_values[validation_mask]),
            torch.from_numpy(bundle.station_indices[validation_mask]),
            torch.from_numpy(normalized_targets[validation_mask]),
        ),
        "test": TensorDataset(
            torch.from_numpy(normalized_values[test_mask]),
            torch.from_numpy(bundle.station_indices[test_mask]),
            torch.from_numpy(normalized_targets[test_mask]),
        ),
    }
    loaders = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=name == "train", num_workers=0)
        for name, dataset in datasets.items()
    }
    device = torch.device("cpu")
    input_size = normalized_values.shape[2]
    model: nn.Module
    if model_name == "lstm":
        model = LSTMForecaster(input_size, len(bundle.station_ids))
    else:
        model = TCNForecaster(input_size, len(bundle.station_ids))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_validation_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, loaders["train"], optimizer, device)
        validation_loss = _run_epoch(model, loaders["validation"], None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("sequence model did not produce a validation checkpoint")
    model.load_state_dict(best_state)

    target_mean = preprocessing["target_log_mean"][0]
    target_scale = preprocessing["target_log_scale"][0]
    split_metrics: dict[str, object] = {}
    station_metrics: dict[str, object] = {}
    for name, mask in (("train", train_mask), ("validation", validation_mask), ("test", test_mask)):
        loader = loaders[name]
        predictions = []
        with torch.no_grad():
            for values, stations, _ in loader:
                predictions.append(model(values.to(device), stations.to(device)).cpu().numpy())
        normalized_prediction = np.concatenate(predictions)
        predicted = np.maximum(0.0, np.expm1(normalized_prediction * target_scale + target_mean))
        actual = bundle.targets[mask]
        persistence = frame.set_index(["station_id", "timestamp_utc"], drop=False)
        station_values_for_rows = np.asarray(bundle.station_ids, dtype=object)[bundle.station_indices[mask]]
        issue_key = pd.MultiIndex.from_arrays(
            [station_values_for_rows, pd.to_datetime(bundle.issue_times[mask], utc=True)],
            names=["station_id", "timestamp_utc"],
        )
        persistence_values = persistence.reindex(issue_key)["pm25_current"].to_numpy()
        valid_persistence = np.isfinite(persistence_values)
        split_metrics[name] = {
            "rows": int(mask.sum()),
            "start_utc": str(pd.Timestamp(bundle.issue_times[mask].min(), tz="UTC")),
            "end_utc": str(pd.Timestamp(bundle.issue_times[mask].max(), tz="UTC")),
            "persistence": _metrics(actual[valid_persistence], persistence_values[valid_persistence]),
            model_name: _metrics(actual, predicted),
        }
        station_rows = {}
        station_values = bundle.station_indices[mask]
        for station_index in sorted(np.unique(station_values)):
            station_mask = station_values == station_index
            valid_station_persistence = station_mask & np.isfinite(persistence_values)
            station_result: dict[str, float | int | None] = {
                "rows": int(station_mask.sum()),
                "persistence_rows": int(valid_station_persistence.sum()),
                "persistence_mae": None,
                f"{model_name}_mae": float(mean_absolute_error(actual[station_mask], predicted[station_mask])),
            }
            if valid_station_persistence.any():
                station_result["persistence_mae"] = float(
                    mean_absolute_error(actual[valid_station_persistence], persistence_values[valid_station_persistence])
                )
            station_rows[bundle.station_ids[station_index]] = station_result
        station_metrics[name] = station_rows

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run_dir / f"{model_name}_{horizon}h.pt"
    metrics_path = run_dir / f"{model_name}_{horizon}h.json"
    torch.save(
        {
            "model_name": model_name,
            "horizon_hours": horizon,
            "sequence_length": sequence_length,
            "station_ids": bundle.station_ids,
            "state_dict": model.state_dict(),
            "preprocessing": preprocessing,
        },
        artifact_path,
    )
    result: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "model_name": model_name,
        "horizon_hours": horizon,
        "sequence_length": sequence_length,
        "stride": stride,
        "split_policy": "train < 2025-07-01; validation < 2025-10-01; test >= 2025-10-01",
        "rows": {name: len(dataset) for name, dataset in datasets.items()},
        "feature_columns": list(SEQUENCE_FEATURES),
        "history": history,
        "splits": split_metrics,
        "station_metrics": station_metrics,
        "artifact": str(artifact_path),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model", choices=("lstm", "tcn"), default="lstm")
    parser.add_argument("--horizon", type=int, choices=(1, 3, 6, 12, 24), default=6)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    result = train(
        args.dataset,
        args.run_dir,
        args.model,
        args.horizon,
        args.sequence_length,
        args.stride,
        args.epochs,
        args.batch_size,
    )
    test = result["splits"]["test"]
    print(
        f"Test {args.model} {args.horizon}h MAE: "
        f"persistence={test['persistence']['mae']:.3f}, "
        f"model={test[args.model]['mae']:.3f}, rows={test['rows']}"
    )


if __name__ == "__main__":
    main()
