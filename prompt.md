# Delhi AirCast completion prompt

You are the lead engineer and research scientist finishing this Delhi AirCast PS-13 project. Work directly in this repository and use `uv` for Python commands and environments. Continue autonomously in small, verifiable phases; do not merely describe work.

## Objective

Turn the current repository into the strongest reproducible Delhi air-quality forecasting project possible: audited multi-source data, leakage-safe features, evaluated models, a reliable local API, clear documentation, and honest completion status.

## Rules

- First inspect Git status, existing manifests, artifacts, scripts, tests, and README. Preserve user changes and never delete raw data or artifacts without explicit approval.
- Use the downloaded datasets intelligently: CPCB/OpenCity, OpenAQ, AirDelhi, weather, CAMS, FIRMS, legacy AQI, Kaggle/reference sources, and live CPCB data. Classify each as primary, auxiliary, validation-only, or excluded, with a reason.
- Keep raw inputs separate and preserve provenance, hashes, time coverage, station identity, units, missingness, duplicates, and licensing notes.
- Prevent leakage rigorously: chronological splits, point-in-time features, future weather only when available at issue time, and identical cohorts for model comparisons.
- Prefer a strong simple model over unnecessary complexity. Compare persistence, XGBoost, and sequence/spatial challengers across 1/3/6/12/24-hour horizons. Promote a model only with evidence across rolling-origin tests and station slices.
- Keep uncertainty, data freshness, missing inputs, model version, and historical/offline status explicit in the API. Never present an offline artifact as live.
- Use `uv run pytest -q`, targeted tests, `git diff --check`, API smoke tests, and reproducible commands after each meaningful phase. Inspect outputs and fix failures before proceeding.
- Keep the project usable on CPU where practical. Avoid downloading duplicate data or rebuilding valid artifacts without a reason.
- Do not expose API keys, credentials, private paths, or secrets in code, logs, manifests, commits, or reports.

## Required workflow

1. Audit the repository and produce a short evidence-based gap list.
2. Verify all dataset manifests and refresh the inventory; repair incomplete or stale processing only where needed.
3. Build or refine a unified, leakage-safe feature contract and document dataset roles.
4. Train, evaluate, and compare the final horizon models with rolling-origin backtests and station-level diagnostics.
5. Investigate regressions and missingness, especially weak stations or horizons, before promotion.
6. Make the FastAPI service load the final artifacts, expose health and forecast endpoints, and return quality/source/model metadata.
7. Add or improve tests for data contracts, leakage boundaries, model artifacts, API behavior, and failure states.
8. Update README, experiment ledger, data inventory, and runbook with exact `uv` commands.
9. Run the complete local gate again and report what is complete, what remains, measured metrics, artifact paths, and any limitations.

Make practical decisions without repeatedly asking for approval, but stop before destructive actions or external publication. At every handoff, show concrete commands, test results, files changed, and evidence for claims.
