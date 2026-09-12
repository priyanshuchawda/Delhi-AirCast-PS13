#!/usr/bin/env python3
"""Render architecture figures for the PS-13 report and slides."""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

NAVY = "#1B3A5F"
ACCENT = "#2C5F8A"
TEAL = "#1F6F64"
GREEN = "#2E7D4F"
AMBER = "#B7791F"
RED = "#B42318"
FILL = "#EEF3F8"
WHITE = "#FFFFFF"
SOFT = "#F7FAFC"


HEADER = """
digraph G {
  graph [
    fontname="Helvetica",
    bgcolor="white",
    pad="0.25",
    ranksep="0.55",
    nodesep="0.35",
    splines="spline",
    outputorder="edgesfirst"
  ];
  node [
    fontname="Helvetica",
    fontsize=11,
    shape=box,
    style="rounded,filled",
    fillcolor="#EEF3F8",
    color="#1B3A5F",
    fontcolor="#1B3A5F",
    penwidth=1.5,
    margin="0.18,0.12"
  ];
  edge [
    fontname="Helvetica",
    fontsize=9,
    color="#5B7A9D",
    penwidth=1.4,
    arrowsize=0.75
  ];
"""


def render_dot(name: str, body: str, extra_header: str = "") -> Path:
    src = FIG / f"{name}.dot"
    png = FIG / f"{name}.png"
    src.write_text(HEADER + extra_header + body + "\n}\n", encoding="utf-8")
    subprocess.run(
        ["dot", "-Tpng", "-Gdpi=180", "-o", str(png), str(src)],
        check=True,
    )
    print("wrote", png)
    return png


def fig_architecture() -> Path:
    body = r"""
  labelloc="t";
  label="Figure. End-to-end Delhi AirCast stack (all free sources)";
  fontsize=13;
  fontcolor="#1B3A5F";

  subgraph cluster_hist {
    label="TRAINING  (once / yearly refresh)";
    color="#1B3A5F";
    style="rounded,filled";
    fillcolor="#F7FAFC";
    fontcolor="#1B3A5F";
    opencity [label="OpenCity CPCB dumps\n39 Delhi stations\n15-min 2024–2025", fillcolor="#E7F6EC"];
    om_arch [label="Open-Meteo archive\nweather (no key)", fillcolor="#E7F6EC"];
    openaq_h [label="OpenAQ historical\n(optional extra years)", fillcolor="#FFF4E5"];
  }

  subgraph cluster_live {
    label="LIVE  (every hour)";
    color="#2C5F8A";
    style="rounded,filled";
    fillcolor="#F7FAFC";
    dgov [label="data.gov.in CPCB API\nofficial snapshot", fillcolor="#E7F6EC"];
    openaq_l [label="OpenAQ live\nbackup", fillcolor="#FFF4E5"];
    om_fc [label="Open-Meteo forecast\nnext 24–48 h weather", fillcolor="#E7F6EC"];
    cams [label="CAMS AQ model\nauxiliary feature only", fillcolor="#FFF4E5"];
  }

  merge [label="Clean · align IST · QC sentinels\nHourly merge  +  lag features", shape=box, fillcolor="#D6E6F5", width=3.6];
  store [label="Feature store\nParquet / DuckDB", fillcolor="#EEF3F8"];
  models [label="Forecast models\nPersistence → XGBoost → LSTM\n(+ optional GNN)", fillcolor="#D6E6F5"];
  aqi [label="CPCB AQI engine\nPM2.5 → category", fillcolor="#E7F6EC"];
  dash [label="Web dashboard\nmap · chart · alerts · staleness", fillcolor="#1B3A5F", fontcolor="#FFFFFF"];

  opencity -> merge;
  om_arch -> merge;
  openaq_h -> merge;
  dgov -> merge;
  openaq_l -> merge;
  om_fc -> merge;
  cams -> merge;
  merge -> store -> models -> aqi -> dash;
"""
    return render_dot("fig01_architecture", body)


def fig_hourly_weekly() -> Path:
    body = r"""
  rankdir=TB;
  labelloc="t";
  label="Figure. Two clocks: hourly forecast vs weekly improve-if-better";
  fontsize=13;
  fontcolor="#1B3A5F";

  subgraph cluster_h {
    label="EVERY HOUR  —  collect and forecast";
    color="#1F6F64";
    style="rounded,filled";
    fillcolor="#F3FAF7";
    h1 [label="Collect CPCB snapshot\n+ weather forecast"];
    h2 [label="Append to live table"];
    h3 [label="Run frozen production model"];
    h4 [label="Write 1–24 h PM2.5 path\n+ AQI + narrative"];
    h1 -> h2 -> h3 -> h4;
  }

  subgraph cluster_w {
    label="EVERY WEEK  —  champion / challenger";
    color="#B7791F";
    style="rounded,filled";
    fillcolor="#FFF8EC";
    w1 [label="Add new labelled hours"];
    w2 [label="Score production model\n(28-day MAE / RMSE)"];
    w3 [label="Retrain candidate"];
    w4 [label="Compare on same window"];
    w5a [label="Better → use the new model", fillcolor="#E7F6EC"];
    w5b [label="Keep the current model", fillcolor="#EEF3F8"];
    w1 -> w2 -> w3 -> w4;
    w4 -> w5a;
    w4 -> w5b;
  }
"""
    return render_dot("fig02_two_clocks", body)


def fig_model_ladder() -> Path:
    body = r"""
  rankdir=LR;
  labelloc="t";
  label="Figure. We start simple, then add a stronger model at each step";
  fontsize=13;
  fontcolor="#1B3A5F";

  m0 [label="0  Persistence\nŷ(t+h) = y(t)\nbaseline", fillcolor="#EEF3F8"];
  m1 [label="1  XGBoost\nlags + weather\n+ calendar", fillcolor="#FFF4E5"];
  m2 [label="2  LSTM\n24 h sequence\n→ 1–24 h path", fillcolor="#D6E6F5"];
  m3 [label="3  Optional GNN\nstations as nodes\nneighbourhood", fillcolor="#E7F6EC"];

  m0 -> m1 [label="then compare"];
  m1 -> m2 [label="keep the better"];
  m2 -> m3 [label="nearby stations"];

  cams [label="CAMS model field\nextra regional\nbackground", fillcolor="#EEF3F8", fontsize=10];
  m0 -> cams [style=dashed, constraint=false];
"""
    return render_dot("fig03_model_ladder", body)


def fig_station_graph() -> Path:
    text = r"""
digraph G {
  graph [fontname="Helvetica", bgcolor="white", pad="0.35", ranksep="0.7", nodesep="0.55",
         labelloc="t", label="Figure. Ten Delhi stations as a neighbourhood graph",
         fontsize=13, fontcolor="#1B3A5F"];
  node [fontname="Helvetica", shape=box, style="rounded,filled", fillcolor="#D6E6F5",
        color="#1B3A5F", fontcolor="#1B3A5F", penwidth=1.5, fontsize=11, margin="0.16,0.10"];
  edge [color="#8AA0B8", penwidth=1.4, arrowhead=none];

  { rank=same; Rohini; "Punjabi Bagh"; ITO; "Anand Vihar"; }
  { rank=same; "NSIT Dwarka"; "Mandir Marg"; "Okhla Ph-2"; }
  { rank=same; "Dwarka S8"; "RK Puram"; }
  { rank=same; "IGI T3"; }

  "Anand Vihar" [fillcolor="#1B3A5F", fontcolor="#FFFFFF"];

  Rohini -> "Punjabi Bagh" -> "NSIT Dwarka" -> "Dwarka S8" -> "IGI T3";
  "Punjabi Bagh" -> "Mandir Marg" -> ITO -> "Anand Vihar";
  "Mandir Marg" -> "RK Puram" -> "Okhla Ph-2" -> "Anand Vihar";
  ITO -> "Okhla Ph-2";
  "NSIT Dwarka" -> "RK Puram" [style=dashed];
}
"""
    src = FIG / "fig04_station_graph.dot"
    png = FIG / "fig04_station_graph.png"
    src.write_text(text, encoding="utf-8")
    subprocess.run(["dot", "-Tpng", "-Gdpi=180", "-o", str(png), str(src)], check=True)
    print("wrote", png)
    return png


def fig_future() -> Path:
    body = r"""
  rankdir=LR;
  labelloc="t";
  label="Figure. Growth path: same stack, wider geography and smarter models";
  fontsize=13;
  fontcolor="#1B3A5F";

  n0 [label="NOW\n10 Delhi stations\nXGBoost + live dashboard", fillcolor="#1B3A5F", fontcolor="#FFFFFF"];
  n1 [label="NEXT\nall 39 Delhi sites\nneighbour lags / GNN", fillcolor="#D6E6F5"];
  n2 [label="NCR\nNoida · Gurugram\nGhaziabad · Faridabad\nsame CPCB API", fillcolor="#D6E6F5"];
  n3 [label="ANY CAAQMS CITY\nPune · Mumbai · Bengaluru\nswap station list", fillcolor="#E7F6EC"];
  n4 [label="PRODUCT\nalerts · public API\nuncertainty bands", fillcolor="#FFF4E5"];

  n0 -> n1 -> n2 -> n3 -> n4;
"""
    return render_dot("fig05_future_growth", body)


def fig_source_layers() -> Path:
    body = r"""
  rankdir=TB;
  labelloc="t";
  label="Figure. Each source has a clear job";
  fontsize=13;
  fontcolor="#1B3A5F";

  subgraph cluster_a {
    label="LABELS  (ground truth PM2.5)";
    color="#2E7D4F";
    fillcolor="#F3FAF7";
    style="rounded,filled";
    a1 [label="OpenCity CPCB concentrations", fillcolor="#E7F6EC"];
    a2 [label="OpenAQ CPCB measurements", fillcolor="#E7F6EC"];
  }
  subgraph cluster_b {
    label="FEATURES";
    color="#2C5F8A";
    fillcolor="#F7FAFC";
    style="rounded,filled";
    b1 [label="Open-Meteo weather", fillcolor="#D6E6F5"];
    b2 [label="calendar / Diwali window", fillcolor="#D6E6F5"];
    b3 [label="CAMS regional AQ  (auxiliary)", fillcolor="#FFF4E5"];
  }
  subgraph cluster_c {
    label="LIVE NOW  (dashboard)";
    color="#1B3A5F";
    fillcolor="#F7FAFC";
    style="rounded,filled";
    c1 [label="1. data.gov.in", fillcolor="#E7F6EC"];
    c2 [label="2. OpenAQ backup", fillcolor="#FFF4E5"];
    c3 [label="3. WAQI optional", fillcolor="#FFF4E5"];
  }
  a1 -> b1 [style=invis];
"""
    return render_dot("fig06_source_layers", body)


def fig_station_map() -> Path:
    stations = {
        "Rohini": (77.1195, 28.7326),
        "Punjabi Bagh": (77.1311, 28.6742),
        "Anand Vihar": (77.3160, 28.6469),
        "ITO": (77.2410, 28.6280),
        "Mandir Marg": (77.2005, 28.6341),
        "NSIT Dwarka": (77.0380, 28.6090),
        "Dwarka S8": (77.0710, 28.5710),
        "RK Puram": (77.1869, 28.5632),
        "IGI T3": (77.1000, 28.5562),
        "Okhla Ph-2": (77.2713, 28.5309),
    }
    # rough NCT box
    fig, ax = plt.subplots(figsize=(10.2, 8.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7FAFC")

    # faint NCT rectangle
    rect = FancyBboxPatch(
        (76.98, 28.48),
        0.42,
        0.32,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=NAVY,
        facecolor="#EEF3F8",
        alpha=0.9,
    )
    ax.add_patch(rect)

    # neighbour lines (schematic)
    links = [
        ("Rohini", "Punjabi Bagh"),
        ("Punjabi Bagh", "NSIT Dwarka"),
        ("NSIT Dwarka", "Dwarka S8"),
        ("Dwarka S8", "IGI T3"),
        ("Punjabi Bagh", "Mandir Marg"),
        ("Mandir Marg", "ITO"),
        ("ITO", "Anand Vihar"),
        ("Mandir Marg", "RK Puram"),
        ("RK Puram", "Okhla Ph-2"),
        ("Okhla Ph-2", "Anand Vihar"),
        ("ITO", "Okhla Ph-2"),
        ("IGI T3", "RK Puram"),
    ]
    for a, b in links:
        x1, y1 = stations[a]
        x2, y2 = stations[b]
        ax.plot([x1, x2], [y1, y2], color="#8AA0B8", linewidth=1.2, zorder=1)

    offsets = {
        "Rohini": (0.012, 0.012),
        "Punjabi Bagh": (-0.09, 0.01),
        "Anand Vihar": (0.012, 0.01),
        "ITO": (0.01, 0.012),
        "Mandir Marg": (-0.095, -0.018),
        "NSIT Dwarka": (-0.10, 0.008),
        "Dwarka S8": (-0.09, -0.018),
        "RK Puram": (0.012, -0.018),
        "IGI T3": (-0.055, -0.022),
        "Okhla Ph-2": (0.012, -0.018),
    }

    for name, (x, y) in stations.items():
        is_av = name == "Anand Vihar"
        ax.scatter(
            [x],
            [y],
            s=220 if is_av else 160,
            c=NAVY if is_av else ACCENT,
            zorder=3,
            edgecolors="white",
            linewidths=1.2,
        )
        dx, dy = offsets[name]
        ax.text(
            x + dx,
            y + dy,
            name,
            fontsize=9,
            color=NAVY,
            fontweight="bold" if is_av else "regular",
            zorder=4,
        )

    ax.text(77.19, 28.785, "DELHI  ·  Phase-1 monitoring panel", fontsize=13, color=NAVY, ha="center", fontweight="bold")
    ax.text(77.19, 28.765, "Ten CPCB / DPCC / IMD sites used for hyperlocal 1–24 h forecasts", fontsize=9, color=ACCENT, ha="center")
    ax.set_xlim(76.95, 77.43)
    ax.set_ylim(28.47, 28.81)
    ax.set_xlabel("Longitude", color=NAVY)
    ax.set_ylabel("Latitude", color=NAVY)
    ax.tick_params(colors=NAVY)
    for spine in ax.spines.values():
        spine.set_color("#C5D0DC")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out = FIG / "fig07_station_map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return out


def fig_dashboard_wireframe() -> Path:
    fig, ax = plt.subplots(figsize=(12.4, 7.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def box(x, y, w, h, fc, ec=NAVY, lw=1.4, rad=0.08):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={rad}", facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(p)

    box(0.25, 6.35, 11.9, 0.65, NAVY, ec=NAVY)
    ax.text(6.2, 6.68, "DELHI AIRCAST   ·   Anand Vihar, Delhi — DPCC    10:00 IST", color="white", ha="center", va="center", fontsize=13, fontweight="bold")

    box(0.25, 3.55, 4.3, 2.6, FILL)
    ax.text(2.4, 5.85, "NOW", color=ACCENT, ha="center", fontsize=9, fontweight="bold")
    ax.text(2.4, 5.35, "AQI  187", color=NAVY, ha="center", fontsize=22, fontweight="bold")
    ax.text(2.4, 4.95, "POOR", color="#C2410C", ha="center", fontsize=12, fontweight="bold")
    ax.text(2.4, 4.45, "PM2.5   145 µg/m³", color=NAVY, ha="center", fontsize=11)
    ax.text(2.4, 4.1, "PM10 220   Temp 27°C   Wind 4 km/h", color=ACCENT, ha="center", fontsize=8)
    ax.text(2.4, 3.75, "Source: CPCB via data.gov.in", color="#667", ha="center", fontsize=7.5, style="italic")

    box(4.7, 3.55, 7.45, 2.6, WHITE)
    ax.text(8.4, 5.9, "PM2.5 FORECAST  (next 12 hours)", color=NAVY, ha="center", fontsize=10, fontweight="bold")
    xs = np.linspace(5.1, 11.6, 8)
    ys = np.array([4.55, 4.7, 4.95, 5.15, 5.05, 4.85, 4.65, 4.5])
    ax.plot(xs, ys, color=ACCENT, lw=2.4)
    ax.scatter(xs, ys, color=NAVY, s=28, zorder=3)
    ax.text(5.1, 4.25, "Now", fontsize=8, color=ACCENT)
    ax.text(8.3, 4.25, "+6 h", fontsize=8, color=ACCENT)
    ax.text(11.2, 4.25, "+12 h", fontsize=8, color=ACCENT)
    ax.text(8.4, 3.75, "Air quality is expected to deteriorate over the next 3 hours.", color=RED, ha="center", fontsize=9, style="italic")

    box(0.25, 0.25, 7.6, 3.1, WHITE)
    ax.text(4.05, 3.05, "DELHI SENSOR MAP", color=NAVY, ha="center", fontsize=10, fontweight="bold")
    # mini dots
    pts = [(1.3, 2.3), (2.2, 1.9), (3.6, 2.15), (4.8, 2.4), (5.6, 1.7), (2.8, 1.2), (4.1, 1.15), (6.4, 2.0), (6.2, 1.2), (1.6, 1.15)]
    names = ["Rohini", "P. Bagh", "Mandir", "ITO", "Anand Vihar", "Dwarka", "RK Puram", "Okhla", "IGI", "NSIT"]
    for (x, y), n in zip(pts, names):
        col = NAVY if n == "Anand Vihar" else ACCENT
        ax.scatter([x], [y], s=90, c=col, zorder=3, edgecolors="white")
        ax.text(x, y - 0.22, n, ha="center", fontsize=6.5, color=NAVY)

    box(8.05, 0.25, 4.1, 3.1, FILL)
    ax.text(10.1, 3.05, "SYSTEM HEALTH", color=NAVY, ha="center", fontsize=10, fontweight="bold")
    lines = [
        "Live feed: CPCB  (21:00 IST)",
        "Weather: Open-Meteo OK",
        "Backup: OpenAQ standby",
        "Model: XGBoost v1  (frozen)",
        "28-day MAE:  tracked continuously",
        "Retrain: weekly, promote if better",
    ]
    for i, line in enumerate(lines):
        ax.text(8.3, 2.6 - 0.38 * i, "•  " + line, fontsize=8.5, color=NAVY, va="center")

    out = FIG / "fig08_dashboard_wireframe.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return out


def main() -> None:
    fig_architecture()
    fig_hourly_weekly()
    fig_model_ladder()
    fig_station_graph()
    fig_future()
    fig_source_layers()
    fig_station_map()
    fig_dashboard_wireframe()
    print("all figures in", FIG)


if __name__ == "__main__":
    main()
