'use client';

import { useState } from 'react';
import useSWR from 'swr';
import {
  Activity, ArrowDownRight, ArrowRight, ArrowUpRight, ChevronDown, CircleHelp,
  Clock3, CloudFog, Compass, ExternalLink, MapPin, RefreshCw,
  ShieldCheck, Wind,
} from 'lucide-react';
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

type StationForecast = {
  station_id: string;
  station_name: string;
  latitude: number | null;
  longitude: number | null;
  as_of_utc: string;
  target_timestamp_utc: string;
  current_pm25: number;
  forecast_pm25: number;
  current_aqi: number | null;
  current_aqi_status: string;
  forecast_aqi: number | null;
  forecast_aqi_category: string | null;
  forecast_aqi_status: string;
  forecast_pollutants: Record<string, number>;
  horizon_hours: number;
  quality_status: string;
};

type ForecastResponse = {
  horizon_hours: number;
  as_of_utc: string | null;
  forecasts: StationForecast[];
};

type HistoryPoint = { timestamp_utc: string; pm25: number };
type HistoryResponse = { history: HistoryPoint[] };

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    const description = response.status === 503 ? 'Forecast artifacts are not installed on the API service.' : `Forecast service returned ${response.status}.`;
    throw new Error(description);
  }
  return response.json() as Promise<T>;
}

const horizons = [1, 3, 6, 12, 24];
const cityBounds = { north: 28.89, south: 28.40, west: 76.84, east: 77.42 };

function aqiTone(value: number | null) {
  if (value === null) return { label: 'Unavailable', color: '#89918f', className: 'tone-muted' };
  if (value <= 50) return { label: 'Good', color: '#32866a', className: 'tone-good' };
  if (value <= 100) return { label: 'Satisfactory', color: '#78a95b', className: 'tone-satisfactory' };
  if (value <= 200) return { label: 'Moderate', color: '#d6a53c', className: 'tone-moderate' };
  if (value <= 300) return { label: 'Poor', color: '#df7548', className: 'tone-poor' };
  if (value <= 400) return { label: 'Very poor', color: '#c9514f', className: 'tone-very-poor' };
  return { label: 'Severe', color: '#8f3a50', className: 'tone-severe' };
}

function prettyTime(value: string | null | undefined) {
  if (!value) return '—';
  return new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata', day: 'numeric', month: 'short', year: 'numeric',
    hour: 'numeric', minute: '2-digit',
  }).format(new Date(value));
}

function shortTime(value: string) {
  return new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata', hour: 'numeric' }).format(new Date(value));
}

function projectPoint(station: StationForecast) {
  const latitude = station.latitude ?? 28.64;
  const longitude = station.longitude ?? 77.21;
  return {
    x: 42 + ((longitude - cityBounds.west) / (cityBounds.east - cityBounds.west)) * 656,
    y: 28 + ((cityBounds.north - latitude) / (cityBounds.north - cityBounds.south)) * 338,
  };
}

function DelhiMap({ stations, selectedId, onSelect }: {
  stations: StationForecast[]; selectedId: string; onSelect: (id: string) => void;
}) {
  return (
    <div className="map-canvas" role="group" aria-label="CPCB air quality monitoring stations across Delhi">
      <svg viewBox="0 0 740 400" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
        <defs>
          <pattern id="map-grid" width="35" height="35" patternUnits="userSpaceOnUse">
            <path d="M 35 0 L 0 0 0 35" fill="none" stroke="#dfe4dc" strokeWidth=".7" />
          </pattern>
          <linearGradient id="map-wash" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0" stopColor="#f2f4ed" />
            <stop offset="1" stopColor="#e8eee5" />
          </linearGradient>
        </defs>
        <rect width="740" height="400" fill="url(#map-wash)" />
        <rect width="740" height="400" fill="url(#map-grid)" />
        <path d="M-12 286 C92 260 100 314 188 285 S320 219 384 251 490 291 564 239 671 228 756 178" fill="none" stroke="#a8c8c5" strokeWidth="13" opacity=".57" />
        <path d="M-12 286 C92 260 100 314 188 285 S320 219 384 251 490 291 564 239 671 228 756 178" fill="none" stroke="#f6fbf7" strokeWidth="2" opacity=".9" />
        <path d="M92 0 L178 400 M358 0 L305 400 M565 0 L522 400 M0 120 L740 90 M0 352 L740 324" fill="none" stroke="#fff" strokeWidth="2" opacity=".75" />
        <path d="M125 70 C222 34 289 72 348 54 S487 69 539 99 622 109 671 90" fill="none" stroke="#d0d8cc" strokeWidth="2" strokeDasharray="5 5" />
        <text x="370" y="195" className="map-city-label">NEW DELHI</text>
        <text x="28" y="34" className="map-neighborhood">NORTH-WEST</text>
        <text x="578" y="365" className="map-neighborhood">SOUTH-EAST</text>
        {stations.filter((station) => station.latitude !== null && station.longitude !== null).map((station) => {
          const { x, y } = projectPoint(station);
          const selected = station.station_id === selectedId;
          const tone = aqiTone(station.forecast_aqi);
          return (
            <g key={station.station_id} className="map-marker" role="button" tabIndex={0}
              aria-label={`${station.station_name}, AQI ${station.forecast_aqi ?? 'unavailable'}`}
              onClick={() => onSelect(station.station_id)}
              onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(station.station_id); } }}>
              {selected && <circle cx={x} cy={y} r="16" fill={tone.color} opacity=".17" />}
              <circle cx={x} cy={y} r={selected ? 7 : 5.3} fill={tone.color} stroke="#fff" strokeWidth={selected ? 2.6 : 1.8} />
              {selected && <text x={x + 12} y={y - 12} className="map-marker-label">{station.station_name.split(',')[0]}</text>}
            </g>
          );
        })}
      </svg>
      <div className="map-compass"><Compass size={15} /><span>N</span></div>
      <div className="map-attribution">Station locations · CPCB / OpenCity</div>
    </div>
  );
}

function MetricCard({ label, value, detail, icon: Icon, accent = false }: {
  label: string; value: string; detail: string; icon: typeof Activity; accent?: boolean;
}) {
  return (
    <article className={`metric-card ${accent ? 'metric-card-accent' : ''}`}>
      <div className="metric-topline"><span>{label}</span><Icon size={17} strokeWidth={1.7} /></div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </article>
  );
}

export default function Home() {
  const [horizon, setHorizon] = useState(6);
  const [selectedId, setSelectedId] = useState('');
  const { data: forecast, error, isLoading: loading, mutate: reloadForecast } = useSWR<ForecastResponse>(
    `/backend/stations/forecast?horizon=${horizon}`,
    fetchJson,
    { revalidateOnFocus: false, shouldRetryOnError: false },
  );
  const stations = forecast?.forecasts ?? [];
  const selected = stations.find((station) => station.station_id === selectedId) ?? stations[0];
  const historyKey = selected ? `/backend/stations/${encodeURIComponent(selected.station_id)}/history?hours=72` : null;
  const { data: historyPayload } = useSWR<HistoryResponse>(historyKey, fetchJson, { revalidateOnFocus: false, shouldRetryOnError: false });
  const history = historyPayload?.history ?? [];

  const tone = aqiTone(selected?.forecast_aqi ?? null);
  const delta = selected ? selected.forecast_pm25 - selected.current_pm25 : 0;
  const measuredAt = selected?.as_of_utc ?? forecast?.as_of_utc ?? null;
  const historyData = history.map((point) => ({ ...point, time: shortTime(point.timestamp_utc) }));
  const pollutants = selected?.forecast_pollutants ?? {};

  return (
    <main className="app-shell">
      <aside className="rail">
        <a className="brand-lockup" href="#top" aria-label="Delhi AirCast home">
          <span className="brand-mark"><Wind size={19} strokeWidth={1.8} /></span>
          <span className="brand-name">AIR<span>CAST</span><small>DELHI · PS-13</small></span>
        </a>
        <div className="rail-divider" />
        <nav className="rail-nav" aria-label="Main navigation">
          <a className="rail-link active" href="#overview"><Activity size={18} /><span>Overview</span></a>
          <a className="rail-link" href="#network"><MapPin size={18} /><span>Monitor network</span></a>
          <a className="rail-link" href="#forecast"><CloudFog size={18} /><span>Forecast</span></a>
        </nav>
        <div className="rail-bottom">
          <div className="rail-note"><ShieldCheck size={16} /><span>Research-grade<br />offline forecast</span></div>
          <span className="rail-version">MODEL · XGBOOST 2.1</span>
        </div>
      </aside>

      <section className="workspace" id="top">
        <header className="topbar">
          <div className="breadcrumb">AIR QUALITY <span>/</span> DELHI NCR</div>
          <div className="topbar-right">
            <span className="archive-pill"><span className="archive-dot" /> HISTORICAL MODE</span>
            <button className="icon-button" aria-label="About forecast data" onClick={() => document.getElementById('data-note')?.scrollIntoView({ behavior: 'smooth' })}><CircleHelp size={18} /></button>
          </div>
        </header>

        <div className="content" id="overview">
          <section className="intro-row">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> AIR QUALITY, A LITTLE CLOSER</div>
              <h1>Know the air<br /><em>before you go.</em></h1>
              <p className="intro-copy">Station-level air quality forecasts across Delhi, built from CPCB observations and a time-aware machine learning model.</p>
            </div>
            <div className="intro-meta">
              <span className="meta-label">LATEST AVAILABLE OBSERVATION</span>
              <strong>{prettyTime(measuredAt)}</strong>
              <span className="meta-note"><Clock3 size={13} /> Historical snapshot · not live</span>
            </div>
          </section>

          {error && <div className="error-banner" role="alert"><CloudFog size={19} /><div><strong>Forecast service unavailable</strong><span>{error.message} Start the FastAPI service and install the local serving bundle.</span></div><button onClick={() => void reloadForecast()}><RefreshCw size={15} /> Retry</button></div>}

          <section className="control-row" aria-label="Forecast controls">
            <label className="control-field station-select"><span className="control-label">MONITORING STATION</span><span className="select-wrap"><MapPin size={16} /><select aria-label="Monitoring station" value={selected?.station_id ?? ''} onChange={(event) => setSelectedId(event.target.value)} disabled={!stations.length}>{stations.map((station) => <option key={station.station_id} value={station.station_id}>{station.station_name}</option>)}</select><ChevronDown size={15} /></span></label>
            <div className="control-field horizon-select"><span className="control-label">FORECAST HORIZON</span><div className="horizon-options" role="group" aria-label="Forecast horizon">{horizons.map((value) => <button key={value} className={horizon === value ? 'horizon active' : 'horizon'} onClick={() => setHorizon(value)} aria-pressed={horizon === value}>{value}<small>h</small></button>)}</div></div>
            <div className="selected-station"><span className="control-label">SELECTED AREA</span><strong><MapPin size={15} /> {selected?.station_name?.split(',')[0] ?? 'Delhi network'}</strong></div>
          </section>

          <section className="metrics-grid" id="forecast" aria-label="Selected station forecast summary">
            <article className={`aqi-card ${tone.className}`}>
              <div className="aqi-card-heading"><span>FORECAST AIR QUALITY INDEX</span><span className="aqi-mini-icon"><CloudFog size={17} /></span></div>
              <div className="aqi-reading"><strong>{loading ? '···' : selected?.forecast_aqi ?? '—'}</strong><span className="aqi-unit">AQI</span></div>
              <div className="aqi-category"><span className="aqi-status-dot" />{selected?.forecast_aqi_category ?? tone.label}<span className="aqi-horizon">IN {horizon} HOURS</span></div>
              <div className="aqi-scale" aria-label="AQI scale"><span /><span /><span /><span /><span /><span /></div>
              <div className="aqi-card-foot"><span>Based on {selected?.forecast_aqi_status === 'cpcb_five_pollutant_subset' ? '5 pollutant forecasts' : 'PM2.5 proxy'}</span><span>0 — 500</span></div>
            </article>
            <MetricCard label="PM₂.₅ FORECAST" value={selected ? `${selected.forecast_pm25.toFixed(1)}` : '—'} detail="µg/m³ · particulate matter" icon={Wind} />
            <MetricCard label="CHANGE FROM NOW" value={selected ? `${delta > 0 ? '+' : ''}${delta.toFixed(1)}` : '—'} detail={selected ? `µg/m³ · ${delta > 0 ? 'expected to rise' : delta < 0 ? 'expected to ease' : 'roughly unchanged'}` : 'Compared with latest reading'} icon={delta > 0 ? ArrowUpRight : ArrowDownRight} />
            <MetricCard label="LATEST MEASURED" value={selected ? selected.current_pm25.toFixed(1) : '—'} detail="µg/m³ · CPCB station reading" icon={Activity} />
          </section>

          <section className="main-grid">
            <article className="panel map-panel" id="network">
              <div className="panel-heading"><div><span className="section-kicker">THE MONITOR NETWORK</span><h2>Delhi, station by station</h2></div><span className="station-count"><span />{stations.length || '—'} CPCB STATIONS</span></div>
              <DelhiMap stations={stations} selectedId={selected?.station_id ?? ''} onSelect={setSelectedId} />
              <div className="map-footer"><div className="map-legend"><span className="legend-title">FORECAST AQI</span>{[['#32866a', 'Good'], ['#78a95b', 'Satisfactory'], ['#d6a53c', 'Moderate'], ['#df7548', 'Poor'], ['#c9514f', 'Very poor'], ['#8f3a50', 'Severe']].map(([color, label]) => <span className="legend-item" key={label}><i style={{ backgroundColor: color }} />{label}</span>)}</div><span className="map-disclaimer">Markers are monitors, not a continuous pollution surface.</span></div>
            </article>

            <div className="side-stack">
              <article className="panel trend-panel">
                <div className="panel-heading compact"><div><span className="section-kicker">RECENT OBSERVATIONS</span><h2>PM₂.₅ · last 72 hours</h2></div><span className="chart-unit">µg/m³</span></div>
                {historyData.length ? <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={historyData} margin={{ top: 10, right: 8, bottom: 0, left: -20 }}><defs><linearGradient id="pm-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#c98440" stopOpacity={0.23} /><stop offset="95%" stopColor="#c98440" stopOpacity={0.015} /></linearGradient></defs><CartesianGrid stroke="#e9e8e1" vertical={false} strokeDasharray="3 5" /><XAxis dataKey="time" tickLine={false} axisLine={false} tick={{ fill: '#8b918c', fontSize: 10 }} interval="preserveStartEnd" /><YAxis tickLine={false} axisLine={false} tick={{ fill: '#8b918c', fontSize: 10 }} /><Tooltip contentStyle={{ border: '1px solid #e4e4dc', borderRadius: 10, fontSize: 12, boxShadow: '0 8px 26px #26322b14' }} formatter={(value) => [`${Number(value).toFixed(1)} µg/m³`, 'PM2.5']} labelFormatter={(_, payload) => payload?.[0]?.payload?.timestamp_utc ? prettyTime(payload[0].payload.timestamp_utc) : ''} /><Area type="monotone" dataKey="pm25" stroke="#bc7940" strokeWidth={2.2} fill="url(#pm-fill)" dot={false} activeDot={{ r: 4, strokeWidth: 0, fill: '#bc7940' }} /></AreaChart></ResponsiveContainer></div> : <div className="chart-empty"><Activity size={18} /><span>Historical station series will appear when the panel is available.</span></div>}
                <div className="chart-footer"><span><i className="chart-key" /> OBSERVED PM₂.₅</span><span>{history.length ? `${history.length} hourly records` : 'Historical panel'}</span></div>
              </article>

              <article className="panel pollutant-panel">
                <div className="panel-heading compact"><div><span className="section-kicker">WHAT SHAPES THE INDEX</span><h2>Pollutant outlook</h2></div><span className="forecast-tag">+{horizon}H</span></div>
                <div className="pollutant-list">{[['pm25', 'PM₂.₅', 'µg/m³'], ['pm10', 'PM₁₀', 'µg/m³'], ['no2', 'NO₂', 'µg/m³'], ['co', 'CO', 'mg/m³'], ['o3', 'O₃', 'µg/m³']].map(([key, label, unit]) => <div className="pollutant-row" key={key}><span className={`pollutant-symbol symbol-${key}`}>{label.replace(/[₀-₉₂₁]/g, '').slice(0, 2)}</span><span className="pollutant-name">{label}</span><strong>{pollutants[key] !== undefined ? pollutants[key].toFixed(key === 'co' ? 2 : 1) : '—'}</strong><small>{unit}</small></div>)}</div>
                <div className="subset-note"><span className="note-mark">i</span><span>{horizon === 6 ? 'Six-hour AQI combines five CPCB pollutant sub-indices. SO₂ and NH₃ are not modelled yet.' : 'The five-pollutant AQI estimate is currently available only at six hours; other horizons use the PM₂.₅ proxy.'}</span></div>
              </article>
            </div>
          </section>

          <section className="bottom-grid">
            <article className="panel forecast-panel">
              <div className="panel-heading compact"><div><span className="section-kicker">PLANNING AHEAD</span><h2>Short-term forecast</h2></div><span className="forecast-caption">Selected station · {selected?.station_name?.split(',')[0] ?? '—'}</span></div>
              <div className="forecast-steps">{horizons.map((step) => { const value = step === horizon ? selected?.forecast_aqi : null; const stepTone = aqiTone(value ?? selected?.forecast_aqi ?? null); return <button className={`forecast-step ${step === horizon ? 'chosen' : ''}`} key={step} onClick={() => setHorizon(step)} aria-label={`Show ${step} hour forecast`}><span className="step-time">+{step}<small>h</small></span><span className="step-line"><i style={{ background: stepTone.color }} /></span><strong>{step === horizon ? (selected?.forecast_aqi ?? '—') : '···'}</strong><small className="step-status">{step === horizon ? stepTone.label : 'Select horizon'}</small></button>; })}</div>
              <div className="forecast-foot"><Clock3 size={14} /> Modelled directly at each horizon from the latest available historical observation.<span>Issued {prettyTime(measuredAt)}</span></div>
            </article>
            <article className="data-note" id="data-note"><div className="data-note-icon"><ShieldCheck size={19} /></div><div><span className="section-kicker">A NOTE ON THIS FORECAST</span><h2>Useful context, honestly shown.</h2><p>This is an offline research model trained on historical CPCB/OpenCity observations with weather, CAMS and fire-history features. It does not use live sensor feeds. The map shows monitored stations only; neighbourhood-wide interpolation is not validated.</p><a href="https://github.com/priyanshuchawda/Delhi-AirCast-PS13/blob/main/EXPERIMENTS.md" target="_blank" rel="noreferrer">Read the model evaluation <ExternalLink size={13} /></a></div></article>
          </section>

          <footer className="page-footer"><span>DELHI AIRCAST <i>·</i> PS-13 AIR QUALITY FORECASTING</span><span>Historical model output · Not a public-health advisory <ArrowRight size={13} /></span></footer>
        </div>
      </section>
    </main>
  );
}
