# Delhi AirCast: production model strategy

Status: implementation blueprint for PS-13  
Scope: Delhi NCT, station-level hourly PM2.5 forecasts for 1–24 hours ahead

This document turns the PS-13 requirements into an engineering and modelling
plan. It is intentionally evidence-driven: every more-complex model is a
candidate, not a promise. The production model is the simplest model that
beats the incumbent on the agreed backtests, important stations, pollution
episodes, and operational reliability.

## 1. Product contract

At issuance time `t`, for station `s`, the system will produce:

- PM2.5 point forecasts at `t+1`, `t+3`, `t+6`, `t+12`, and `t+24` hours;
- an hourly path for the complete 1–24 hour window;
- 80% and 90% prediction intervals;
- a PM2.5-derived CPCB AQI view, clearly labelled as a PM2.5-only proxy when
  the other pollutant inputs required for official AQI are unavailable;
- forecast age, data-quality status, feature/model version, and a short trend
  explanation such as improving, stable, or deteriorating.

The API and dashboard must never imply that a stale forecast is live. Every
response therefore carries `issued_at`, `data_as_of`, `horizon_hours`,
`station_id`, `model_version`, `interval_method`, and `quality_status`.

The official AQI path is separate from the PM2.5 forecasting target. When the
required pollutant observations or pollutant forecasts are present, calculate
the official CPCB composite according to the project’s AQI implementation.
Otherwise expose the PM2.5 sub-index/proxy and say so in the UI.

## 2. Data contract and leakage rules

### Label

The supervised label is the quality-controlled hourly CPCB PM2.5 value:

```text
y(s, t, h) = PM2.5 at station s at t + h
```

The initial model should use the 39-station CPCB 2024–2025 panel, with the
PS-13 8–10 station subset as the first product slice. Expansion to all stations
must be a configuration change, not a new modelling architecture.

### Point-in-time feature rule

For an issue time `t`, a feature may be used only if it would have been known
at `t`. This applies to every source:

- observed station pollution and weather: values timestamped no later than `t`;
- weather forecasts: the forecast vintage issued by `t`, not the later realised
  weather value;
- CAMS/Open-Meteo air-quality fields: only the available forecast/analysis
  vintage, with its publication time retained;
- FIRMS: only detections published by `t`; historical fire counts must not use
  a later reprocessed file as if it was available in real time;
- OpenAQ, AirDelhi, Kaggle, and other derivatives: useful for comparison,
  pretraining experiments, or coverage analysis, but never silently mixed into
  the CPCB label without provenance and timestamp checks.

This is the most important correctness rule. A random split or a join to
future weather can produce an impressive but unusable score.

### Quality and provenance

The canonical panel will include:

- station identity, coordinates, aliases, timezone, and active date ranges;
- raw value, cleaned value, source timestamp, ingestion timestamp, and source;
- missingness, imputation, and stale-value flags;
- quality flags for CPCB sentinels (`999`, `998`, `985`, `1999`), negative
  values, unit conversions, and physically implausible extremes;
- station-month coverage and duplicate/conflict counts;
- a dataset manifest containing source URLs, retrieval time, row counts,
  checksums, and transformation version.

Short carry-forward may be used for input features (maximum three hours, with
an explicit flag). Missing future labels remain missing; they are not imputed
for evaluation.

## 3. Feature system

Features are grouped so ablation results remain interpretable.

### A. Target history

- PM2.5 lags: 1, 2, 3, 6, 12, 24, 48, and 72 hours;
- rolling mean, median, standard deviation, minimum, maximum, and slope over
  3, 6, 12, 24, and 72 hours;
- same-hour previous day and previous week;
- recent PM10, NO2, SO2, CO, O3, and NH3 where the station has trustworthy
  observations;
- explicit missingness and age-of-last-observation features.

### B. Time and calendar

- sine/cosine hour-of-day and day-of-week;
- month, season, weekend, holiday, school/exam and known event windows;
- sunrise/sunset or daylight indicators;
- Diwali and crop-burning windows as event features, not as causal claims.

### C. Local meteorology

- temperature, relative humidity, dew point, pressure, wind speed/direction;
- wind represented as u/v components rather than only compass categories;
- precipitation, cloud cover, boundary-layer or mixing-height proxies where
  available;
- weather lags, rolling changes, and missingness flags;
- point-in-time Open-Meteo forecast fields for future hours when available.

### D. Spatial pollution information

- k-nearest station PM2.5 lag/rolling summaries;
- distance-weighted neighbour values;
- upwind/downwind summaries using wind direction;
- neighbour freshness and count of available stations;
- station embedding or static station metadata for a global model.

The first spatial model is engineered neighbour features. A learned graph is
added only after this baseline has been measured.

### E. Regional and event signals

- CAMS regional PM2.5/PM10/NO2 and aerosol fields as auxiliary predictors;
- FIRMS fire count, FRP sum/max, distance bands, directional/upwind counts,
  and 6/12/24/72-hour rolling values;
- rainfall and wind-cleansing indicators;
- optional traffic/emissions proxies only when the timestamp and licensing
  permit point-in-time use.

All feature groups support ablation: target history only, +time, +weather,
+spatial, +CAMS, +FIRMS. This tells us what is genuinely useful for Delhi.

## 4. Evaluation design

### Splits

No random train/test split is permitted for the product score. We will use
rolling-origin, blocked backtests:

1. train on an earlier period, validate on the next block;
2. roll the origin forward and repeat across seasons;
3. reserve the latest complete period as a final untouched test;
4. maintain a recent live-like holdout after every data refresh.

The initial 2024–2025 experiment can use several origins spanning winter,
summer, monsoon, and post-monsoon. A candidate is evaluated at all stations,
not just the easiest station.

### Metrics

Primary:

- PM2.5 MAE, reported overall and at each horizon;
- PM2.5 RMSE for spike sensitivity;
- skill versus persistence and same-hour-previous-day baselines.

Product metrics:

- PM2.5-derived AQI MAE and category accuracy/balanced F1;
- threshold-crossing accuracy for CPCB AQI categories;
- high-pollution slice MAE and underprediction rate;
- interval coverage and average interval width at 80% and 90%;
- p50/p95 inference latency and successful-response rate.

Required slices:

- station and station cluster;
- horizon 1/3/6/12/24;
- hour of day and season;
- winter inversion, rain, dust, Diwali, and fire-influence windows;
- clean versus polluted periods;
- complete versus stale/missing input conditions.

We will publish a scorecard with sample counts. A metric without its slice
count is not considered evidence.

## 5. Model ladder

Every rung uses the same feature snapshots, splits, metrics, and manifest.

### Rung 0: sanity baselines

1. Last observed PM2.5 (persistence).
2. Same hour yesterday.
3. Same hour last week.
4. Rolling and seasonal combinations.
5. CAMS-only and weather-only reference models.

These establish whether the ML system adds value at all. The current single-
station baseline is promising but is not yet a product result.

### Rung 1: global tabular workhorse

Train one model across stations and horizons with station metadata/embedding
or one-hot station identity. Start with HistGradientBoosting and then test
XGBoost or LightGBM if available. Use direct horizon heads or one model per
horizon rather than recursively feeding predictions back into itself.

This is the most likely production winner because it handles nonlinear weather,
missing flags, event features, and engineered neighbour information with low
latency and straightforward error analysis. XGBoost is a recognised gradient-
boosted tree method, but its use here is justified by the backtest rather than
by its name.

### Rung 2: spatial feature model

Add the neighbour/upwind features above, with strict point-in-time joins. Test
fixed distance graphs and k-nearest graphs. If this does not beat the global
workhorse, retain the simpler model and record the negative result.

### Rung 3: sequence models

Test a compact GRU/LSTM and a temporal convolutional network using 48–168 hours
of history and a direct multi-horizon decoder. A Temporal Fusion Transformer
is a later challenger because it explicitly separates static, known-future,
and observed inputs and can produce quantiles. It is useful only if the data
volume and operational complexity justify it.

### Rung 4: graph neural forecasters

Test STGCN and DCRNN-style models on the station graph. The graph can begin as
distance plus prevailing-wind edges, then be compared with a learned/adaptive
adjacency. Keep the graph small and auditable. With roughly 39 stations, a
graph model is feasible, but not automatically superior; the station graph is
not a road network and airflow is direction- and regime-dependent.

### Rung 5: calibrated ensemble

If different models win in different regimes, blend only on validation data:

- persistence for very short horizons or stale inputs;
- boosting for normal station-level forecasts;
- sequence/graph model for spatial or episode slices.

Weights are frozen before the final test. The ensemble must beat the best
single candidate and must not hide a severe regression at a critical station.

## 6. Uncertainty and safe behaviour

The user needs to know when the forecast is uncertain. We will implement:

1. quantile models for p10/p50/p90 at every horizon;
2. rolling, horizon-specific conformal calibration on a held-out calibration
   window;
3. non-crossing interval checks and clipping only at physically meaningful
   lower bounds;
4. a quality state: `live`, `degraded`, `stale`, or `unavailable`;
5. fallback to persistence with a clear degraded badge when required inputs or
   the model are unavailable.

Conformalized quantile regression is a suitable calibration layer because it
combines quantile forecasts with conformal residual calibration; interval
coverage still has to be measured on Delhi’s time-ordered backtests.

## 7. Promotion gates

The following are proposed engineering gates, adjustable after the first full
backtest:

- beat persistence on aggregate MAE by at least 5% at 1/6/12/24 hours;
- no critical station or season regression greater than 5% without an explicit
  product decision;
- reduce high-pollution underprediction, not only average error;
- AQI category performance must not regress while PM2.5 MAE improves;
- 80% intervals should achieve approximately 80% empirical coverage and 90%
  intervals approximately 90%, with width reported alongside coverage;
- p95 inference latency and memory fit the service budget;
- reproducible artifact from a clean checkout with a data manifest;
- API response includes provenance, freshness, quality state, and model SHA.

The incumbent remains live until all gates pass. A failed challenger is still
valuable if its scorecard and failure slices are preserved.

## 8. Production architecture

```text
source connectors
    -> immutable raw data + manifests
    -> canonical station/time panel
    -> QC and point-in-time feature snapshots
    -> train/evaluate/promote pipeline
    -> versioned model + feature schema
    -> FastAPI inference service
    -> dashboard/map and monitoring
```

Each model artifact records:

- git commit and configuration;
- training and calibration time ranges;
- source manifests and checksums;
- feature schema and missing-value policy;
- model parameters, metrics, and slice scorecard;
- AQI conversion version;
- rollback predecessor.

Serving and offline evaluation must call the same feature-building code. A
separate notebook-only transformation is not acceptable.

Monitoring will track data freshness, missingness, station coverage, feature
drift, forecast distribution, interval coverage after labels arrive, and
rolling 28-day error. Retraining is triggered by schedule plus evidence, for
example sustained error degradation, not by an unreviewed automatic overwrite.
The previous model remains available for rollback.

## 9. Implementation sequence

The practical PR sequence is:

1. canonical station registry, schemas, provenance, and data-quality report;
2. multi-station hourly panel and point-in-time feature snapshots;
3. rolling-origin evaluation harness and all baseline scorecards;
4. global multi-horizon boosting model;
5. neighbour/upwind spatial features and ablation;
6. quantile forecasts and conformal calibration;
7. GRU/LSTM/TCN challenger, only if the harness is stable;
8. STGCN/DCRNN challenger and graph ablations;
9. ensemble selection and promotion gates;
10. live scheduler, FastAPI contract, dashboard, monitoring, and rollback;
11. reproducibility run from a clean checkout and final public documentation.

The current dataset collection is already broad enough to begin steps 1–4.
OpenAQ and FIRMS can continue in parallel, but they should enter the model
only through measured ablations and point-in-time availability checks. More
datasets are not a substitute for a correct label panel and honest evaluation.

## 10. Research and design references

Project requirements and initial scope: `Delhi_AirCast_PS13_Data_Acquisition_Report.pdf`.

Primary technical references:

- [STGCN, IJCAI 2018](https://www.ijcai.org/proceedings/2018/505) — graph and temporal convolution for spatiotemporal forecasting.
- [DCRNN, ICLR 2018](https://openreview.net/pdf?id=SJiHXGWAZ) — diffusion-style graph recurrence and multi-step forecasting.
- [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363) — multi-horizon forecasting with static, known-future, observed inputs, and interpretability.
- [Conformalized Quantile Regression](https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf) — quantile prediction with conformal calibration.
- [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api) and [air-quality/CAMS API](https://open-meteo.com/en/docs/air-quality-api).
- [CPCB real-time air-quality resource](https://www.data.gov.in/resource/real-time-air-quality-index-various-locations).
- [NASA FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/) for the optional fire-influence feature family.

## Bottom line

The best real project is not “all datasets plus the deepest model.” It is a
traceable, station-wide, multi-horizon system in which:

1. CPCB remains the label authority;
2. every feature is available at issue time;
3. persistence is a respected incumbent;
4. boosting plus spatial features is the first serious production candidate;
5. sequence and graph models are tested fairly and retained only when they
   improve difficult slices;
6. uncertainty, staleness, AQI semantics, and rollback are product features;
7. every claim is backed by a reproducible rolling backtest.

