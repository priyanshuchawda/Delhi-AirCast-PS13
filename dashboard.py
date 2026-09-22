"""Offline Streamlit dashboard for the Delhi AirCast project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from scripts.train_multistation_baseline import _feature_frame


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/processed/delhi_unified_forecast.parquet"
RUN_DIR = ROOT / "data/runs/final_xgboost"
HORIZONS = (1, 3, 6, 12, 24)


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

with st.expander("Model and dataset details"):
    st.write({
        "station": station_id,
        "rows in unified table": f"{len(frame):,}",
        "stations": len(stations),
        "coverage": f"{frame['timestamp_utc'].min():%Y-%m-%d} → {frame['timestamp_utc'].max():%Y-%m-%d}",
        "feature set": feature_set,
        "model": "offline XGBoost final artifact",
        "sources": "CPCB/OpenCity, Open-Meteo weather, CAMS, NASA FIRMS",
    })

st.caption("Forecasts are historical/offline research artifacts; they are not presented as current live measurements.")
