# Delhi AirCast — PS-13

College report + review slides for **PS-13 AI-Powered Air Quality Forecasting**.

## Deliverables

| File | What |
|------|------|
| `Delhi_AirCast_PS13_Data_Acquisition_Report.docx` | Full report (edit cover: name, roll, college) |
| `Delhi_AirCast_PS13_Data_Acquisition_Report.pdf` | Printable PDF |
| `Delhi_AirCast_PS13_Presentation.pptx` | Review deck |
| `Delhi_AirCast_PS13_Presentation.pdf` | Slides as PDF |
| `figures/` | Architecture diagrams (Graphviz + Matplotlib) |

## Regenerate

```bash
cd ~/Delhi-AirCast-PS13
uv run python generate_diagrams.py
uv run python generate_report.py
uv run python generate_ppt.py
soffice --headless --convert-to pdf:writer_pdf_Export Delhi_AirCast_PS13_Data_Acquisition_Report.docx
soffice --headless --convert-to pdf Delhi_AirCast_PS13_Presentation.pptx
```

Fill the cover-page blanks before submitting.

## Acquire public datasets

Downloaded data is intentionally excluded from Git. Keep the raw files under
`data/raw/` and retain the generated manifests and source metadata.

```bash
uv run python scripts/download_datasets.py --source all
```

The downloader currently collects the 78-resource OpenCity/CPCB Delhi archive,
central-Delhi historical weather, CAMS air-quality history, and CPCB reference
documents. Large research mirrors are downloaded separately when their public
hosting requires a dedicated client.

Normalize the downloaded AirDelhi hourly spatial grid with:

```bash
uv run python scripts/normalize_airdelhi.py
```

This writes an ignored Parquet artifact under `data/processed/` and a local
manifest containing source hashes, time coverage, duplicate handling, and the
output hash.

Build and evaluate a first one-hour-ahead CPCB baseline:

```bash
uv run python scripts/build_forecast_dataset.py --station-id site_105
uv run python scripts/train_baseline.py
```

The baseline uses chronological train/validation/test partitions and reports
both persistence and gradient-boosted predictions. Forecast features are
constructed from information available at the prediction timestamp; no random
shuffle is used.

Analyze the complete CPCB station panel and build the multi-station feature
table:

    uv run python scripts/analyze_cpcb_panel.py
    uv run python scripts/build_multistation_dataset.py

The quality report records coverage, missingness, duplicate timestamps, PM2.5
statistics, network availability, and conservative coordinate matches to the
live CPCB snapshot. The feature builder writes direct PM2.5 targets for
1/3/6/12/24 hours and uses only current or past observations.

Train the first global multi-station XGBoost challenger:

    uv run python scripts/train_multistation_baseline.py --horizon 1
    uv run python scripts/train_multistation_baseline.py --horizon 6

The measured results and promotion decision are tracked in
EXPERIMENTS.md.

Run expanding-window quarterly backtests:

    uv run python scripts/evaluate_backtests.py --horizon 1 --model persistence
    uv run python scripts/evaluate_backtests.py --horizon 1 --model xgboost --n-estimators 120 --n-jobs 6

Train CPU sequence challengers and compare them with the same test cohort:

    uv run python scripts/train_sequence_model.py --model lstm --horizon 6 --sequence-length 48 --stride 12 --epochs 3
    uv run python scripts/train_sequence_model.py --model tcn --horizon 6 --sequence-length 48 --stride 12 --epochs 3
    uv run python scripts/compare_sequence_cohort.py --model tcn --horizon 6 --sequence-length 48 --stride 12

Train the final offline-serving models using the maximum leakage-safe unified
feature table. This fits each horizon on all observations before the held-out
test tail and writes the model artifacts and evaluation summary under
`data/runs/final_xgboost/`:

```bash
uv run python scripts/train_final_models.py --n-jobs 4
```

The local API prefers these final offline artifacts for 1/3/6/12/24-hour
forecasts. The service intentionally reports them as historical/offline
forecasts; no live-data dependency is required for this project version.

Acquire OpenAQ Delhi/NCR batches with a verified key kept outside the project:

```bash
export OPENAQ_API_KEY='your-64-character-openaq-key'
uv run python scripts/download_openaq.py --batch catalog
uv run python scripts/download_openaq.py --batch latest
uv run python scripts/download_openaq.py --batch hourly \
  --start 2026-08-01T00:00:00Z --end 2026-09-01T00:00:00Z
```

The OpenAQ downloader stores raw catalog, latest-reading, and hourly-average
responses under `data/raw/openaq/`. It defaults to reference-monitor locations
and PM10, PM2.5, O3, CO, NO2, and SO2. Historical batches are resumable and
record response counts, hashes, and provenance in JSON manifests.

Normalize downloaded OpenAQ hourly batches into Parquet with:

```bash
uv run python scripts/normalize_openaq.py
```

The output is written to `data/processed/openaq/` with a matching manifest.

Download NASA FIRMS active-fire observations for Delhi and the North India
upwind region with a MAP_KEY kept outside the repository:

```bash
export FIRMS_MAP_KEY='your-firms-map-key'
uv run python scripts/download_firms.py \
  --start-date 2024-01-01 \
  --end-date 2025-12-31
```

The default bounding box is `74,25,79.5,31.5`; it covers Delhi/NCR and
important Punjab, Haryana, Rajasthan, and Uttar Pradesh upwind areas. The
connector uses five-day windows, saves raw CSV files and SHA-256 manifests
under `data/raw/firms/`, retries transient failures, and resumes completed
windows. NASA FIRMS is an auxiliary fire-event signal, not a replacement for
CPCB station observations.

Acquire the official CPCB live Delhi snapshot with a data.gov.in API key kept
outside the repository:

```bash
export DATAGOV_API_KEY='your-data-gov-api-key'
uv run python scripts/download_cpcb_live.py --state Delhi
```

The connector stores raw paginated JSON, a long-form Parquet snapshot, and a
manifest under `data/raw/cpcb_live/`. It resumes completed pages by default.

Normalize the downloaded OpenCity 2017–2023 AQI-only archive separately with:

```bash
uv run python scripts/normalize_cpcb_legacy_aqi.py
```

This produces `data/processed/cpcb_2017_2023_aqi.parquet`. It is retained as
an AQI validation/history layer and is not mixed into pollutant-concentration
training targets without a separate audit.

Run the local API after building the station artifact:

```bash
uv run uvicorn delhi_aircast.api:app --reload
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/aqi \\
  -H 'content-type: application/json' \\
  -d '{"pm25":145}'
curl http://127.0.0.1:8000/forecast/site_105

When the multi-station feature table and XGBoost artifacts exist, the forecast
endpoint automatically prefers them:

    curl 'http://127.0.0.1:8000/forecast/site_105?horizon=6'
    curl http://127.0.0.1:8000/forecast/site_105/path

Responses include issue time, target time, model artifact, horizon, feature
count, and an explicit historical/offline quality state. This prevents the
offline research panel from being presented as a live forecast until live
feature ingestion is implemented.
```

Launch the offline dashboard:

```bash
uv run streamlit run dashboard.py
```

The dashboard reads the saved unified feature table and final model artifacts
locally. It provides station history, offline PM2.5 forecasts for 1/3/6/12/24
hours, model quality metrics, and dataset provenance without requiring a live
data connection.

Optional comparison sources are listed in [`KAGGLE_DATASETS.md`](KAGGLE_DATASETS.md).
They can be downloaded with `uv run python scripts/download_kaggle.py` after
installing the external client with `uv tool install kaggle`.

## Validate the AQI engine

```bash
uv run pytest -q
```

The CPCB implementation returns a clearly labelled PM2.5-only proxy when a
station does not have enough pollutant inputs for an official overall AQI. It
does not silently turn a single PM2.5 measurement into a full AQI.
