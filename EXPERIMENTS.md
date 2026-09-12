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

### Rolling-origin 1-hour evaluation

Command:

    uv run python scripts/evaluate_backtests.py --horizon 1 --model persistence --output data/runs/rolling_persistence_1h.json
    uv run python scripts/evaluate_backtests.py --horizon 1 --model xgboost --n-estimators 120 --n-jobs 6 --output data/runs/rolling_xgboost_1h.json

The evaluator uses five expanding-window quarterly test periods. Mean fold MAE
was 16.383 for persistence and 14.466 for XGBoost, an 11.7% improvement.
XGBoost improved 176 of 195 station-fold slices and regressed 19. The largest
regression is still site_113 in the latest high-pollution quarter, so the model
remains a challenger rather than the promoted API model.

## Next experiments

1. Add rolling-origin backtests across seasons and report slice counts.
2. Train and compare 3/6/12/24-hour direct horizons.
3. Investigate site_113 and other station-level regressions.
4. Compare no-neighbour, geographic-neighbour, and wind-aware feature sets.
5. Add quantile forecasts and conformal calibration.
6. Only then benchmark TCN/LSTM and graph challengers against the same splits.

### Sequence challengers, 6-hour horizon

Commands:

    uv run python scripts/train_sequence_model.py --model lstm --horizon 6 --sequence-length 48 --stride 12 --epochs 3 --batch-size 256
    uv run python scripts/train_sequence_model.py --model tcn --horizon 6 --sequence-length 48 --stride 12 --epochs 3 --batch-size 256
    uv run python scripts/compare_sequence_cohort.py --model tcn --horizon 6 --sequence-length 48 --stride 12

The LSTM and TCN use 48 hours of history, train-only normalization, explicit
missingness masks, station embeddings, and direct 6-hour targets. On the
identical 6,708-row test cohort with valid issue-time persistence:

| Model | MAE | RMSE |
|---|---:|---:|
| Persistence | 66.812 | 94.043 |
| Global XGBoost | 42.814 | 63.032 |
| LSTM | 45.195 | 69.235 |
| TCN | 45.197 | 68.380 |

The tree model wins this controlled comparison by about 5.3% MAE. The sequence
models are retained as reproducible challengers, not promoted. A larger stride
or fewer epochs is suitable for CPU prototyping, but final comparisons must use
the same cohort and a rolling-origin protocol.
