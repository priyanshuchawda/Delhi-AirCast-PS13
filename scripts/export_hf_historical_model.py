#!/usr/bin/env python3
"""Prepare a clean Hugging Face model repo with historical XGBoost weights.

Only trained model artifacts, their measured evaluation summary, and a model
card are exported. Training datasets and API credentials are never copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 3, 6, 12, 24)


def build_model_card(metrics: dict[str, object]) -> str:
    horizons = metrics["horizons"]
    table = [
        "| Horizon | Test rows | Persistence MAE | XGBoost MAE | MAE gain | XGBoost RMSE | XGBoost R² |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        result = horizons[str(horizon)]
        baseline = result["persistence"]["mae"]
        score = result["xgboost"]["mae"]
        gain = (baseline - score) / baseline * 100
        table.append(
            f"| {horizon} h | {result['test_rows']:,} | {baseline:.2f} µg/m³ "
            f"| {score:.2f} µg/m³ | {gain:.1f}% | "
            f"{result['xgboost']['rmse']:.2f} µg/m³ | {result['xgboost']['r2']:.3f} |"
        )

    return "\n".join(
        [
            "---",
            "library_name: xgboost",
            "pipeline_tag: tabular-regression",
            "tags:",
            "  - air-quality",
            "  - pm25",
            "  - time-series-forecasting",
            "  - delhi",
            "  - xgboost",
            "  - cpcb",
            "---",
            "",
            "# Delhi AirCast: historical Delhi PM2.5 forecast models",
            "",
            "Five direct-horizon XGBoost regressors forecast station-level PM2.5 "
            "concentration (µg/m³) for 1, 3, 6, 12, and 24 hours ahead. The models "
            "were trained from historical Delhi CPCB/OpenCity observations with "
            "time-aware pollutant, weather, atmospheric-composition, fire-history, "
            "and spatial-neighbour features.",
            "",
            "> **Historical research artifacts, not a live service.** These weights "
            "do not fetch current observations or weather. Their forecasts are only "
            "as current as the input feature row supplied at inference time. They "
            "forecast PM2.5 concentration, not official CPCB composite AQI.",
            "",
            "## Artifacts",
            "",
            "| File | Forecast horizon |",
            "|---|---:|",
            *[f"| `model_{h}h.joblib` | {h} hour(s) |" for h in HORIZONS],
            "| `evaluation_metrics.json` | Chronological test and persistence-baseline metrics |",
            "",
            "Each Joblib artifact stores the fitted XGBoost regressor, its ordered "
            "feature-column contract, horizon, station encoding, and fit metadata.",
            "",
            "## Evaluation",
            "",
            "The panel contains 684,216 station-hour rows for 39 Delhi stations "
            "from 2024-01-01 through 2025-12-31. Models use a chronological split: "
            "eligible observations before 2025-10-01 for fitting, July–September "
            "2025 as the validation interval for model development, and "
            "2025-10-01 through 2025-12-31 as the reported chronological test tail. "
            "The final artifacts are refit on all eligible pre-test data. The test "
            "tail was used in prior project model comparisons; treat these scores "
            "as project evaluation results, not a pristine independent benchmark.",
            "",
            *table,
            "",
            "MAE/RMSE are in PM2.5 concentration units (µg/m³); R² is unitless. "
            "Persistence predicts the future concentration as the latest available "
            "PM2.5 value at issue time. See `evaluation_metrics.json` for exact "
            "counts and split metadata.",
            "",
            "## Intended use and limitations",
            "",
            "- Research, coursework, reproducibility, and offline model comparisons "
            "for Delhi station-level short-term PM2.5 forecasting.",
            "- Not a health advisory, regulatory measurement, or neighbourhood-wide "
            "interpolation product.",
            "- The historical evaluation period does not establish accuracy on "
            "current conditions or at unmonitored locations.",
            "- To predict, construct a point-in-time feature row using the same "
            "feature engineering and station identifiers used for training. The "
            "model weights alone do not supply those observations/features.",
            "- One station (site_106) did not have a verified coordinate match in "
            "the source station registry; do not infer its location from this model.",
            "",
            "## Loading a model",
            "",
            "Install compatible `joblib`, `pandas`, `scikit-learn`, and `xgboost` "
            "versions. Joblib uses Python pickle serialization: only load artifacts "
            "from a source you trust.",
            "",
            "```python",
            "import joblib",
            "import pandas as pd",
            "",
            "artifact = joblib.load(\"model_1h.joblib\")",
            "feature_columns = artifact[\"feature_columns\"]",
            "# Build this row with the project's point-in-time feature pipeline.",
            "feature_row = {name: ... for name in feature_columns}",
            "X = pd.DataFrame([feature_row], columns=feature_columns)",
            "pm25_1h = float(artifact[\"model\"].predict(X)[0])",
            "print(f\"1-hour PM2.5 forecast: {pm25_1h:.1f} µg/m³\")",
            "```",
            "",
            "## Data and provenance",
            "",
            "Primary training panel: [OpenCity Delhi Hourly Air Quality Reports]"
            "(https://data.opencity.in/dataset/delhi-hourly-air-quality-reports). "
            "Weather and auxiliary feature sources and processing are documented "
            "in the [project data-source inventory]"
            "(https://github.com/priyanshuchawda/Delhi-AirCast-PS13/blob/main/DATA_SOURCES.md) "
            "and [experiment ledger]"
            "(https://github.com/priyanshuchawda/Delhi-AirCast-PS13/blob/main/EXPERIMENTS.md). "
            "This model repository contains no training dataset.",
            "",
            "No blanket model license is asserted here. Check upstream data-source "
            "terms and obtain appropriate permission before redistribution or "
            "commercial use.",
            "",
            "Project: [Delhi AirCast PS-13]"
            "(https://github.com/priyanshuchawda/Delhi-AirCast-PS13) · "
            "Dashboard: [Delhi AirCast]"
            "(https://delhi-aircast-ps13.vercel.app).",
            "",
        ]
    )


def export(output: Path) -> dict[str, object]:
    source = ROOT / "data/runs/final_xgboost"
    evaluation_path = source / "final_evaluation.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"Historical evaluation file not found: {evaluation_path}")
    metrics = json.loads(evaluation_path.read_text(encoding="utf-8"))
    missing = [h for h in HORIZONS if not (source / f"model_{h}h.joblib").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing historical horizon artifacts: {missing}")

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for horizon in HORIZONS:
        shutil.copy2(source / f"model_{horizon}h.joblib", output / f"model_{horizon}h.joblib")

    # Keep only the auditable split and score summary; no rows, source data,
    # user-specific paths, API keys, or per-row predictions are exported.
    summary = {
        "generated_at": metrics["generated_at"],
        "model_family": "XGBoost regressor (one direct model per horizon)",
        "feature_set": metrics["feature_set"],
        "target": "future station PM2.5 concentration, micrograms per cubic metre",
        "source_panel": {
            "rows": 684216,
            "stations": 39,
            "period": ["2024-01-01", "2025-12-31"],
        },
        "fit_policy": "final models refit on eligible rows before the held-out test interval",
        "test_interval": ["2025-10-01T00:00:00Z", "2025-12-31T22:00:00Z"],
        "test_tail_caveat": "This interval was used in prior project model comparisons; not an untouched independent benchmark.",
        "horizons": {
            str(h): {
                key: metrics["horizons"][str(h)][key]
                for key in ("fit_rows", "test_rows", "test_interval", "persistence", "xgboost")
            }
            for h in HORIZONS
        },
    }
    (output / "evaluation_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(build_model_card(summary), encoding="utf-8")

    return {
        "output": str(output),
        "files": sorted(path.name for path in output.iterdir()),
        "bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
        "training_rows": summary["source_panel"]["rows"],
        "horizons": list(HORIZONS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".hf-historical-model",
        help="Output directory containing only model weights, metrics, and card",
    )
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))


if __name__ == "__main__":
    main()
