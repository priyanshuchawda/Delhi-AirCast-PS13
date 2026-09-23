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

### Data inventory checkpoint

Command:

    uv run python scripts/inventory_sources.py

At the latest checkpoint the project contained 6,956 inventoried files totaling
5.58 GB:

| Area | Files | Size |
|---|---:|---:|
| Kaggle raw | 470 | 1,950.4 MiB |
| OpenAQ raw | 4,769 | 1,242.3 MiB |
| Hugging Face raw | 191 | 973.7 MiB |
| OpenCity raw | 80 | 450.2 MiB |
| AirDelhi raw | 37 | 346.2 MiB |
| FIRMS raw | 1,321 | 97.6 MiB |
| processed artifacts | 58 | 215.5 MiB |

The OpenAQ downloader is still running, so this is a checkpoint rather than a
final inventory. The inventory script excludes partial/lock files and records
SHA-256 hashes for completed files.

### Spatial GNN challenger across forecast horizons

The spatial graph challenger treats each coordinate-matched CPCB station as a
node and connects each node to its six nearest stations. Node features include
recent PM2.5, weather, and time-of-day values. Inputs are normalized using the
training period; PM2.5 targets use train-fitted standardized `log1p`. Model
selection uses the Jul–Sep 2025 validation interval and early stopping. The
Oct–Dec 2025 test tail is held for final scoring. Each row compares the GNN,
saved final XGBoost, and persistence on the same available station-hour rows
for that horizon.

| Horizon | Test rows | Persistence MAE | Spatial GNN MAE | Final XGBoost MAE | GNN AQI-category accuracy | XGBoost AQI-category accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1h | 80,224 | 22.879 | 21.802 | 18.887 | 78.7% | 81.2% |
| 3h | 79,664 | 48.141 | 37.979 | 35.339 | 65.2% | 68.2% |
| 6h | 79,254 | 70.650 | 49.393 | 44.246 | 57.9% | 61.7% |
| 12h | 78,780 | 85.446 | 58.956 | 50.901 | 50.2% | 57.8% |
| 24h | 78,257 | 58.672 | 61.145 | 56.117 | 48.6% | 54.7% |

AQI and category metrics use the project's explicitly labelled PM2.5-only AQI
proxy, not an official multi-pollutant composite. The GNN improves on
persistence through 12 hours but loses to XGBoost at every horizon and loses to
persistence at 24 hours. XGBoost therefore remains the selected model. The GNN
is retained as a reproducible spatial challenger; these scores do not justify
moving its training to a remote GPU. Metrics and checkpoints are generated
under `data/runs/spatial_gnn/` and are local artifacts, not checked into Git.

Reproduce the six-hour experiment with:

    uv run python scripts/train_spatial_gnn.py --horizon 6 --epochs 20 --lr 0.001 --patience 5

### Dashboard model comparison

For the selected station, issue hour, and horizon, the offline dashboard scores
all compatible saved artifacts together: final XGBoost, persistence, the
spatial GNN where its full graph input is available, and 48-hour LSTM/TCN at
the six-hour horizon. It reports PM2.5 and its explicitly-labelled AQI proxy
and category for each. An optional simple mean of learned-model predictions is
shown as experimental and unvalidated; it does not replace XGBoost. Artifacts
that are absent or lack compatible inputs are omitted rather than synthesized.

### Feature-family and model-strategy diagnostic

`uv run python scripts/compare_model_strategies.py --horizons 1 6 24 --n-jobs 4`
compares CPCB sensor/time/neighbour features, incremental weather and CAMS
families, and the all-source CPCB + Open-Meteo + CAMS + FIRMS feature set.
It also compares XGBoost with scikit-learn histogram gradient boosting. The
diagnostic uses a deterministic one-in-three training-row sample per station
to keep CPU cost bounded; it retains every validation/test row and uses the
same chronological windows for each candidate. Blend weight is selected only
on Jul–Sep 2025 validation MAE; Oct–Dec remains the held-out test interval.

Six-hour findings from the local run:

| Features / model | Validation PM2.5 MAE | Held-out PM2.5 MAE |
|---|---:|---:|
| CPCB sensor/time/neighbour features, XGBoost | 11.834 | 47.930 |
| + weather, XGBoost | 12.117 | 45.618 |
| + weather and CAMS, XGBoost | 12.291 | 44.547 |
| + FIRMS (all sources), XGBoost | 12.906 | 44.283 |
| All sources, HistGradientBoosting |  | 44.587 |
| Validation-selected XGBoost/HistGradientBoosting blend |  | 44.283 |

The summer validation distribution is much easier than the polluted Oct–Dec
test tail, so its low absolute errors should not be read as expected winter
performance. In this sampled six-hour comparison, auxiliary feature groups
improve the polluted test MAE by 3.647 µg/m³ versus CPCB-only features, while
the more complex tree blend is rejected by validation (it selects XGBoost
alone). This supports retaining the existing all-source XGBoost; it does not
justify promoting a neural model or a blend. Final serving artifacts remain
trained on all available pre-test rows, not the diagnostic subsample.

Across all evaluated horizons, auxiliary data does not help uniformly. At one
hour, CPCB-only XGBoost scores 19.376 µg/m³ held-out PM2.5 MAE versus 19.595
for all sources; at 24 hours, all-source XGBoost scores 54.741 versus 54.618
for CPCB-only. At six hours, weather, CAMS, and FIRMS reduce CPCB-only MAE
from 47.930 to 44.283. The 24-hour validation-selected blend gains only
0.010 µg/m³ over XGBoost on test (54.608 vs 54.618), too little to justify
another serving model. Keep all-source XGBoost as the practical six-hour
choice; do not claim every feature family helps at every horizon.

### Multi-pollutant six-hour CPCB AQI estimate

The PM2.5-only forecast does not satisfy the multi-pollutant AQI part of the
problem statement, so this experiment trains separate six-hour XGBoost models
for PM10, NO2, CO, and O3 and combines them with the existing PM2.5 model using
the project's CPCB sub-index/AQI calculation. The extra models are fitted on
all eligible pre-test rows (target timestamps are purged at the split); the
Oct–Dec 2025 test tail was already used in earlier candidate comparisons, so
these are internal historical results, not an untouched external benchmark.

On the identical 81,074-row test cohort where PM2.5 is available, the
five-pollutant subset AQI estimate scores MAE 47.10 and category accuracy
62.10%. A PM2.5-only sub-index proxy scores MAE 51.68 and 57.77% category
accuracy against the same observed composite AQI: a 4.58-point MAE reduction
and 4.33 percentage-point category-accuracy gain. AQI persistence scores MAE
75.97 and 47.18% category accuracy on that cohort. This is a measured
improvement, but not a complete all-pollutant forecast: SO2 and NH3 are
omitted, AQI is calculated from the available forecast subset, and the
dashboard serves this model only at six hours. Per-pollutant scores, cohort
rules, and artifact paths are recorded in the ignored local
`data/runs/multipollutant_aqi_final/evaluation_6h.json`.

Train/rebuild the CPU artifacts with:

    uv run python scripts/train_multipollutant_aqi.py --horizon 6 --run-dir data/runs/multipollutant_aqi_final

Artifacts and evaluation outputs remain local under `data/runs/`; they are
excluded from Git along with source datasets.

OpenAQ is not pooled into CPCB labels. A preliminary nearest-coordinate
cross-source check found 47,649 same-hour overlapping PM2.5 pairs across 35
stations, but about 80.7 µg/m³ MAE and only 0.50 overall correlation. The
station-by-station spread is large. Until timestamp aggregation and monitor
identity are reconciled, that data is useful for a data-quality investigation,
not as interchangeable supervised truth or a clean external score.
