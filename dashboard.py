"""Offline Streamlit dashboard for the Delhi AirCast project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import torch
from torch import nn

from src.delhi_aircast.aqi import calculate_aqi, pm25_proxy
from scripts.train_multistation_baseline import _feature_frame
from scripts.train_sequence_model import LSTMForecaster, TCNForecaster, SEQUENCE_FEATURES
from scripts.train_spatial_gnn import FEATURES as GNN_FEATURES, SpatialGNN


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/processed/delhi_unified_forecast.parquet"
RUN_DIR = ROOT / "data/runs/final_xgboost"
MULTIPOLLUTANT_RUN_DIR = ROOT / "data/runs/multipollutant_aqi_final"
REGISTRY = ROOT / "data/processed/cpcb_panel_quality/station_registry.csv"
HORIZONS = (1, 3, 6, 12, 24)
AQI_POLLUTANTS = ("pm25", "pm10", "no2", "co", "o3")


st.set_page_config(page_title="Delhi AirCast", page_icon="🌫️", layout="wide")


@st.cache_data(show_spinner="Loading the offline feature table…")
def load_dataset() -> pd.DataFrame:
    frame = pd.read_parquet(DATASET)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame.sort_values(["station_id", "timestamp_utc"])


@st.cache_resource(show_spinner="Preparing model features…")
def prepare_features(frame: pd.DataFrame):
    return _feature_frame(frame.copy())


@st.cache_resource
def load_bundle(horizon: int) -> dict:
    return joblib.load(RUN_DIR / f"model_{horizon}h.joblib")


@st.cache_resource
def load_torch_bundle(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


@st.cache_resource
def load_pollutant_bundle(pollutant: str, horizon: int) -> dict:
    if pollutant == "pm25":
        return load_bundle(horizon)
    return joblib.load(MULTIPOLLUTANT_RUN_DIR / f"{pollutant}_{horizon}h.joblib")


def multipollutant_forecast(features: pd.DataFrame, row_index: int, horizon: int):
    """Return a CPCB AQI estimate only when the full tested pollutant set exists."""
    if not all(
        (MULTIPOLLUTANT_RUN_DIR / f"{pollutant}_{horizon}h.joblib").exists()
        for pollutant in AQI_POLLUTANTS
        if pollutant != "pm25"
    ):
        return None
    concentrations: dict[str, float] = {}
    for pollutant in AQI_POLLUTANTS:
        bundle = load_pollutant_bundle(pollutant, horizon)
        vector = features.loc[[row_index], bundle["feature_columns"]].astype(float)
        concentrations[pollutant] = max(0.0, float(bundle["model"].predict(vector)[0]))
    return concentrations, calculate_aqi(concentrations)


def model_comparison(frame: pd.DataFrame, station_id: str, horizon: int, issue_time: pd.Timestamp) -> pd.DataFrame:
    """Score every compatible saved model for the same station and issue hour."""
    results: list[dict[str, object]] = []
    station_rows = frame[(frame["station_id"].astype(str) == station_id) & (frame["timestamp_utc"] == issue_time)]
    if station_rows.empty:
        return pd.DataFrame()
    source_index = station_rows.index[0]
    current = float(station_rows.iloc[0]["pm25_current"])

    def add(name: str, value: float, note: str = "") -> None:
        value = max(0.0, float(value))
        proxy = pm25_proxy(value)
        results.append({"Model": name, "Forecast PM2.5": value, "AQI proxy": proxy.value, "Proxy category": proxy.category, "Note": note})

    features, _, _ = prepare_features(frame)
    xgb = load_bundle(horizon)
    vector = features.loc[[source_index], xgb["feature_columns"]].astype(float)
    add("XGBoost (selected)", xgb["model"].predict(vector)[0], "Primary model")
    add("Persistence", current, "Current PM2.5 carried forward")

    gnn_path = ROOT / f"data/runs/spatial_gnn/model_{horizon}h.pt"
    if gnn_path.exists():
        artifact = load_torch_bundle(str(gnn_path))
        ids = artifact["station_ids"]
        at_time = frame[frame["timestamp_utc"] == issue_time].copy()
        at_time["station_id"] = at_time["station_id"].astype(str)
        rows = at_time.drop_duplicates("station_id").set_index("station_id")
        if all(station in rows.index for station in ids):
            raw = rows.loc[ids, list(GNN_FEATURES)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            means = np.asarray(artifact["means"], dtype=np.float32)
            scales = np.asarray(artifact["scales"], dtype=np.float32)
            raw = np.where(np.isfinite(raw), raw, means)
            normalized = torch.tensor((raw - means) / scales, dtype=torch.float32)
            graph = torch.tensor(artifact["adjacency"], dtype=torch.float32)
            model = SpatialGNN(len(GNN_FEATURES), 48, graph)
            model.load_state_dict(artifact["state_dict"])
            model.eval()
            with torch.no_grad():
                output = model(normalized)[ids.index(station_id)].item()
                add("Spatial GNN", np.expm1(output * artifact["target_scale"] + artifact["target_mean"]), "Research challenger")

    if horizon == 6:
        ordered = frame[frame["station_id"].astype(str) == station_id].sort_values("timestamp_utc")
        ordered = ordered[ordered["timestamp_utc"] <= issue_time].tail(48)
        if len(ordered) == 48 and (ordered["timestamp_utc"].diff().dropna() == pd.Timedelta(hours=1)).all():
            for name in ("lstm", "tcn"):
                path = ROOT / f"data/runs/sequence_models/{name}_6h.pt"
                if not path.exists():
                    continue
                artifact = load_torch_bundle(str(path))
                if station_id not in artifact["station_ids"]:
                    continue
                prep = artifact["preprocessing"]
                raw = ordered[list(SEQUENCE_FEATURES)].to_numpy(dtype=np.float32)
                missing = np.isnan(raw).astype(np.float32)
                raw = np.where(np.isnan(raw), np.asarray(prep["medians"], dtype=np.float32), raw)
                normalized = (raw - np.asarray(prep["means"], dtype=np.float32)) / np.asarray(prep["scales"], dtype=np.float32)
                values = torch.tensor(np.concatenate([normalized, missing], axis=1)[None], dtype=torch.float32)
                station_index = torch.tensor([artifact["station_ids"].index(station_id)])
                model = LSTMForecaster(values.shape[2], len(artifact["station_ids"])) if name == "lstm" else TCNForecaster(values.shape[2], len(artifact["station_ids"]))
                model.load_state_dict(artifact["state_dict"])
                model.eval()
                with torch.no_grad():
                    log_prediction = model(values, station_index).item()
                add(name.upper(), np.expm1(log_prediction * prep["target_log_scale"][0] + prep["target_log_mean"][0]), "Sequence challenger")

    learned = [row["Forecast PM2.5"] for row in results if row["Model"] not in ("Persistence", "XGBoost (selected)")]
    learned.append(results[0]["Forecast PM2.5"])
    if len(learned) > 1:
        add("Experimental model mean", np.mean(learned), "Unvalidated; simple mean of available learned models")
    return pd.DataFrame(results)


@st.cache_data(show_spinner="Preparing the station forecast map…")
def station_forecasts(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY)
    features, _, _ = prepare_features(frame)
    bundle = load_bundle(horizon)
    rows: list[dict] = []
    for station_id, station in frame.groupby(frame["station_id"].astype(str)):
        usable = station.dropna(subset=["pm25_current"])
        if usable.empty:
            continue
        index = usable["timestamp_utc"].idxmax()
        vector = features.loc[[index], bundle["feature_columns"]].astype(float)
        forecast_pm25 = max(0.0, float(bundle["model"].predict(vector)[0]))
        current_pm25 = float(frame.loc[index, "pm25_current"])
        multi = multipollutant_forecast(features, index, horizon)
        proxy = pm25_proxy(forecast_pm25)
        aqi_value = multi[1].value if multi and multi[1].value is not None else proxy.value
        aqi_category = multi[1].category if multi and multi[1].value is not None else proxy.category
        aqi_type = "5-pollutant CPCB estimate" if multi and multi[1].value is not None else "PM2.5-only proxy"
        rows.append({
            "station_id": station_id,
            "current_pm25": current_pm25,
            "forecast_pm25": forecast_pm25,
            "current_aqi": pm25_proxy(current_pm25).value,
            "forecast_aqi": aqi_value,
            "forecast_category": aqi_category,
            "forecast_aqi_type": aqi_type,
            "issue_time": frame.loc[index, "timestamp_utc"],
        })
    scored = pd.DataFrame(rows)
    return scored.merge(registry[["station_id", "station_name", "latitude", "longitude"]], on="station_id", how="left").dropna(subset=["latitude", "longitude"])


def forecast(frame: pd.DataFrame, station_id: str, horizon: int) -> tuple[float, pd.Timestamp, str]:
    feature_frame, _, _ = prepare_features(frame)
    bundle = load_bundle(horizon)
    station = frame[frame["station_id"].astype(str) == station_id]
    if station.empty:
        raise ValueError("Station is not available in the unified dataset")
    usable = station.index[station["pm25_current"].notna()]
    if len(usable) == 0:
        raise ValueError("No complete offline feature row is available for this station")
    latest_index = station.loc[usable, "timestamp_utc"].idxmax()
    row = feature_frame.loc[[latest_index], bundle["feature_columns"]].astype(float)
    value = max(0.0, float(bundle["model"].predict(row)[0]))
    issue_time = frame.loc[latest_index, "timestamp_utc"]
    return value, issue_time, bundle["feature_set"]


def metric_gain(summary: dict) -> float:
    persistence = summary["persistence"]["mae"]
    model = summary["xgboost"]["mae"]
    return (persistence - model) / persistence * 100


st.title("Delhi AirCast")
st.caption("Offline PM2.5 forecasting dashboard · CPCB/OpenCity + weather + CAMS + FIRMS")

if not DATASET.exists():
    st.error(f"Feature table not found: {DATASET}")
    st.stop()

frame = load_dataset()
stations = sorted(frame["station_id"].astype(str).unique())
station_id = st.sidebar.selectbox("Station", stations, index=stations.index("site_105") if "site_105" in stations else 0)
horizon = st.sidebar.select_slider("Forecast horizon (hours)", options=list(HORIZONS), value=6)
st.sidebar.info("This dashboard uses saved offline artifacts. It does not fetch live data.")

station = frame[frame["station_id"].astype(str) == station_id].copy()
latest = station.dropna(subset=["pm25_current"]).iloc[-1]
latest_pm25 = float(latest["pm25_current"])

try:
    prediction, issue_time, feature_set = forecast(frame, station_id, horizon)
except Exception as exc:
    st.error(f"Unable to score the saved model: {exc}")
    st.stop()

left, middle, right = st.columns(3)
left.metric("Latest PM2.5", f"{latest_pm25:.1f} µg/m³")
middle.metric(f"Forecast in {horizon}h", f"{prediction:.1f} µg/m³")
right.metric("Forecast issue time", issue_time.strftime("%Y-%m-%d %H:%M UTC"))

feature_frame, _, _ = prepare_features(frame)
issue_index = station.index[station["timestamp_utc"].eq(issue_time)][0]
multi_forecast = multipollutant_forecast(feature_frame, issue_index, horizon)
forecast_aqi = pm25_proxy(prediction)
if multi_forecast and multi_forecast[1].value is not None:
    concentrations, composite = multi_forecast
    st.info(f"Six-hour CPCB AQI estimate: **{composite.value} ({composite.category})** · Forecast sub-indices from PM₂.₅, PM₁₀, NO₂, CO and O₃.")
    st.caption("This uses the available five-pollutant forecast subset; SO₂ and NH₃ are not forecast in this model version. It is an offline historical forecast, not current live air quality.")
    st.dataframe(pd.DataFrame([{"Pollutant": p.upper(), "Forecast concentration": value, "CPCB sub-index": composite.subindices.get(p)} for p, value in concentrations.items()]), hide_index=True, width="stretch")
else:
    st.info(f"Forecast AQI proxy: **{forecast_aqi.value} ({forecast_aqi.category})** · Based on predicted PM₂.₅ only; official CPCB AQI needs at least three valid pollutant sub-indices including PM₂.₅ or PM₁₀.")

st.subheader("Model predictions for this station and issue time")
comparison = model_comparison(frame, station_id, horizon, issue_time)
if not comparison.empty:
    st.dataframe(comparison.style.format({"Forecast PM2.5": "{:.1f}"}), hide_index=True, use_container_width=True)
    st.caption("Models score the same station and issue hour. The mean is an exploratory blend, not a separately trained or validated model; the selected forecast remains XGBoost.")
else:
    st.caption("No compatible saved model artifacts are available for this selection.")

st.subheader(f"Delhi station forecast map · +{horizon} hours")
map_frame = station_forecasts(frame, horizon)
map_frame["color"] = map_frame["forecast_aqi"].map(lambda value: [0, 180, 80] if value <= 100 else [255, 190, 0] if value <= 200 else [255, 90, 0] if value <= 300 else [180, 30, 60])
map_frame["radius"] = map_frame["station_id"].eq(station_id).map({True: 140, False: 85})
st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(latitude=28.64, longitude=77.21, zoom=10.2, pitch=0),
    layers=[pdk.Layer(
        "ScatterplotLayer",
        data=map_frame,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.85,
    )],
    tooltip={"html": "<b>{station_name}</b><br/>Current PM2.5: {current_pm25}<br/>Forecast PM2.5: {forecast_pm25}<br/>{forecast_aqi_type}: {forecast_aqi} ({forecast_category})"},
), height=500)
st.caption("Map markers show monitored CPCB stations; they are not a validated continuous neighbourhood-level pollution surface.")

st.subheader("Recent station history")
history = station.set_index("timestamp_utc")["pm25_current"].dropna().tail(168).rename("PM2.5")
st.line_chart(history, y="PM2.5", height=300)

st.subheader("Model quality")
cols = st.columns(len(HORIZONS))
for col, h in zip(cols, HORIZONS):
    result_path = RUN_DIR / f"result_{h}h.json"
    if not result_path.exists():
        col.warning(f"{h}h unavailable")
        continue
    result = json.loads(result_path.read_text(encoding="utf-8"))
    col.metric(f"{h}h MAE", f"{result['xgboost']['mae']:.1f}", f"{metric_gain(result):+.1f}% vs persistence")

st.subheader("Spatial model validation")
gnn_rows = []
for h in HORIZONS:
    comparison_path = ROOT / f"data/runs/spatial_gnn/model_{h}h.json"
    if not comparison_path.exists():
        continue
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    gnn_rows.append({
        "Horizon": f"{h}h",
        "Test station-hours": comparison["test_rows"],
        "Persistence MAE": comparison["persistence"]["mae"],
        "Spatial GNN MAE": comparison["metrics"]["mae"],
        "XGBoost MAE (same rows)": comparison["xgboost_same_cohort"]["mae"],
        "GNN AQI category accuracy": comparison["aqi_proxy"]["category_accuracy"],
        "XGBoost AQI category accuracy": comparison["xgboost_aqi_proxy_same_cohort"]["category_accuracy"],
    })
if gnn_rows:
    st.dataframe(pd.DataFrame(gnn_rows).set_index("Horizon").style.format({
        "Persistence MAE": "{:.1f}",
        "Spatial GNN MAE": "{:.1f}",
        "XGBoost MAE (same rows)": "{:.1f}",
        "GNN AQI category accuracy": "{:.1%}",
        "XGBoost AQI category accuracy": "{:.1%}",
    }), use_container_width=True)
    st.caption("The GNN is a spatial challenger. XGBoost remains selected because it has lower test MAE at all horizons; at 24h, GNN also trails persistence.")
else:
    st.caption("Train the spatial challenger to show its same-cohort comparison here.")

with st.expander("Model and dataset details"):
    st.write({
        "station": station_id,
        "rows in unified table": f"{len(frame):,}",
        "stations": len(stations),
        "coverage": f"{frame['timestamp_utc'].min():%Y-%m-%d} → {frame['timestamp_utc'].max():%Y-%m-%d}",
        "feature set": feature_set,
        "model": "offline XGBoost primary; available sequence/GNN challengers shown below",
        "sources": "CPCB/OpenCity, Open-Meteo weather, CAMS, NASA FIRMS",
    })

st.caption("Forecasts are historical/offline research artifacts; they are not presented as current live measurements.")
