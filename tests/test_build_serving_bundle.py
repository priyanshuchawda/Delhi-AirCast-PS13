import importlib.util
from pathlib import Path

import joblib
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_serving_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_serving_bundle", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_bundle_contains_only_latest_features_station_history_and_models(tmp_path):
    data_root = tmp_path / "source"
    output = tmp_path / "bundle"
    dataset_path = data_root / "data/processed/delhi_unified_forecast.parquet"
    dataset_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "station_id": ["site_1", "site_1", "site_2"],
            "station_name": ["North", "North", "South"],
            "timestamp_utc": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T01:00Z", "2025-01-01T00:00Z"]),
            "pm25_current": [10.0, 20.0, 30.0],
            "unused_target": [999.0, 999.0, 999.0],
        }
    ).to_parquet(dataset_path, index=False)
    registry_path = data_root / "data/processed/cpcb_panel_quality/station_registry.csv"
    registry_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"station_id": ["site_1", "site_2"], "station_name": ["North", "South"], "latitude": [28.6, 28.7], "longitude": [77.1, 77.2]}
    ).to_csv(registry_path, index=False)
    model_path = data_root / "data/runs/final_xgboost/model_1h.joblib"
    model_path.parent.mkdir(parents=True)
    joblib.dump({"feature_columns": ["pm25_current", "station_site_1"]}, model_path)

    manifest = module.build_bundle(data_root, output)

    compact = pd.read_parquet(output / "data/processed/delhi_serving_features.parquet")
    history = pd.read_parquet(output / "data/processed/delhi_station_history.parquet")
    assert manifest["stations"] == 2
    assert compact.set_index("station_id").loc["site_1", "pm25_current"] == 20.0
    assert "unused_target" not in compact
    assert len(history) == 3
    assert (output / "data/runs/final_xgboost/model_1h.joblib").exists()
