# Delhi AirCast experiment ledger

This file records measured experiments and promotion decisions. Generated
Parquet files and model binaries remain local under data/, while the
reproducible commands and results stay reviewable in Git.

## 2026-09-12 — CPCB panel quality and first global model

### Dataset audit

Command:

    uv run python scripts/analyze_cpcb_panel.py

Measured result:

- 39 CPCB/OpenCity station files
- 684,216 hourly rows
- 2024-01-01T00:00:00Z through 2025-12-31T23:00:00Z
- 93.20% mean network hourly PM2.5 availability
- no station below 60% structural hourly coverage
- 38/39 station names matched to live CPCB coordinates
- one station (site_106, IGI Airport) remains unmatched and is not assigned a guessed coordinate

The report is generated at data/processed/cpcb_panel_quality/ and includes
station quality, station registry, network hourly coverage, and a JSON summary.

### Point-in-time feature table

Command:

    uv run python scripts/build_multistation_dataset.py

The table contains current and lagged station observations, rolling PM2.5
statistics, calendar features, current nearest-station summaries, and direct
targets at 1/3/6/12/24 hours. It does not join realised future weather or
future pollution values into issue-time features.

### Global XGBoost, 1-hour horizon

Command:

    uv run python scripts/train_multistation_baseline.py --horizon 1

Split:

- train: before 2025-07-01
- validation: 2025-07-01 through 2025-09-30
- test: 2025-10-01 onward

Held-out test result:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence | 22.692 | 37.143 | 0.891 |
| Global XGBoost | 18.527 | 29.853 | 0.929 |

This is an 18.3% aggregate MAE improvement over persistence. It is not yet a
promotion because site_113 regressed from 7.956 to 12.006 MAE. The next
experiment must investigate station-specific calibration/features and rolling
origins before this model is used by the API.

Artifacts are generated under data/runs/multistation_xgboost/.

## Next experiments

1. Add rolling-origin backtests across seasons and report slice counts.
2. Train and compare 3/6/12/24-hour direct horizons.
3. Investigate site_113 and other station-level regressions.
4. Compare no-neighbour, geographic-neighbour, and wind-aware feature sets.
5. Add quantile forecasts and conformal calibration.
6. Only then benchmark TCN/LSTM and graph challengers against the same splits.
