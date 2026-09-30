#!/usr/bin/env python3
"""Build a fact-checked, presentation-ready PS-13 project report."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Delhi_AirCast_PS13_Project_Report.docx"
FIG_DIR = ROOT / "figures"
FIGURE = FIG_DIR / "project_pipeline_overview.png"
REPORT_DATE = date(2026, 10, 1)

NAVY = "18324B"
BLUE = "2E74B5"
PALE_BLUE = "E8EEF5"
PALE_GOLD = "FFF4E5"
INK = "202A34"
MUTED = "5D6B78"
GRID = "CBD5DF"
WHITE = "FFFFFF"
GREEN = "2F6F54"


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def parquet_shape(path: str) -> tuple[int, int]:
    metadata = pq.ParquetFile(ROOT / path).metadata
    return metadata.num_rows, metadata.num_columns


def set_font(run, *, name="Calibri", size=11, color=INK, bold=False, italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    run.bold = bold
    run.italic = italic


def set_cell_shading(cell, color: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), color)
    shading.set(qn("w:val"), "clear")


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_borders(cell, color=GRID, size="4"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_table_geometry(table, widths_dxa: list[int], *, indent_dxa=120):
    total = sum(widths_dxa)
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        column = OxmlElement("w:gridCol")
        column.set(qn("w:w"), str(width))
        grid.append(column)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            cell.width = Inches(widths_dxa[index] / 1440)
            tc_w = cell._tc.get_or_add_tcPr().find(qn("w:tcW"))
            tc_w.set(qn("w:w"), str(widths_dxa[index]))
            tc_w.set(qn("w:type"), "dxa")
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            set_cell_borders(cell)


def add_table(doc, headers, rows, widths_dxa, *, font_size=9, header_fill=PALE_BLUE):
    table = doc.add_table(rows=1, cols=len(headers))
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05
        set_font(p.add_run(str(value)), size=font_size, bold=True, color=NAVY)
        set_cell_shading(cell, header_fill)
    header_pr = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_pr.append(repeat)

    for row_data in rows:
        row = table.add_row()
        row_properties = row._tr.get_or_add_trPr()
        row_cannot_split = OxmlElement("w:cantSplit")
        row_properties.append(row_cannot_split)
        cells = row.cells
        for cell, value in zip(cells, row_data):
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            set_font(p.add_run(str(value)), size=font_size, color=INK)
    set_table_geometry(table, widths_dxa)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_para(doc, text="", *, size=11, color=INK, bold=False, italic=False,
             align=WD_ALIGN_PARAGRAPH.LEFT, after=6, before=0, keep=False):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.10
    p.paragraph_format.keep_together = keep
    if text:
        set_font(p.add_run(text), size=size, color=color, bold=bold, italic=italic)
    return p


def add_mixed_para(doc, pieces, *, after=6, before=0, size=11):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.10
    for text, options in pieces:
        set_font(p.add_run(text), size=size, **options)
    return p


def add_heading(doc, text: str, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(text), size={1: 16, 2: 13, 3: 12}[level],
             color={1: BLUE, 2: BLUE, 3: "1F4D78"}[level], bold=True)
    return p


def add_callout(doc, label, text, *, fill=PALE_BLUE):
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    set_cell_borders(cell, color=fill, size="0")
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.08
    set_font(p.add_run(label), size=10, color=NAVY, bold=True)
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(0)
    p2.paragraph_format.line_spacing = 1.08
    set_font(p2.add_run(text), size=10.5, color=INK)
    set_table_geometry(table, [9360], indent_dxa=120)
    add_para(doc, after=1)


def add_hyperlink(paragraph, text, url):
    relationship_id = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLUE)
    props.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(underline)
    run.append(props)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_page_field(paragraph):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    shown = OxmlElement("w:t")
    shown.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for item in (begin, instruction, separate, shown, end):
        run._r.append(item)
    set_font(run, size=9, color=MUTED)


def style_document(doc):
    for section in doc.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        section.header_distance = Inches(0.492)
        section.footer_distance = Inches(0.492)

        header = section.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.LEFT
        header.paragraph_format.space_after = Pt(0)
        set_font(header.add_run("DELHI AIRCAST  /  PS-13 PROJECT REPORT"), size=8.5, color=MUTED, bold=True)

        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        footer.paragraph_format.space_before = Pt(0)
        footer.paragraph_format.space_after = Pt(0)
        set_font(footer.add_run("Delhi AirCast  |  "), size=9, color=MUTED)
        add_page_field(footer)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, "1F4D78", 8, 4),
    ):
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True


def create_pipeline_figure():
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    nodes = [
        (0.15, "CPCB / OpenCity\n39 Delhi stations"),
        (2.35, "QC + hourly panel\n684,216 rows"),
        (4.55, "Time-safe features\nweather / CAMS / fire"),
        (6.75, "XGBoost heads\n1, 3, 6, 12, 24 h"),
        (8.95, "Forecast + AQI\nAPI / dashboard"),
    ]
    for x, label in nodes:
        box = plt.Rectangle((x, 0.72), 1.8, 1.05, facecolor="#E8EEF5", edgecolor="#2E74B5", linewidth=1.2, zorder=2)
        ax.add_patch(box)
        ax.text(x + 0.9, 1.245, label, ha="center", va="center", fontsize=9, color="#18324B", weight="semibold")
    for x in (1.98, 4.18, 6.38, 8.58):
        ax.annotate("", xy=(x + 0.32, 1.245), xytext=(x, 1.245), arrowprops={"arrowstyle": "->", "color": "#5D6B78", "lw": 1.5})
    ax.text(5.5, 0.28, "Separate live pilot: WAQI current PM₂.₅ index → two-station HistGradientBoosting forecast", ha="center", fontsize=8.5, color="#5D6B78")
    fig.tight_layout(pad=0.2)
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_mae_figure(horizons):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = [int(h) for h in horizons]
    persistence = [horizons[str(h)]["persistence"]["mae"] for h in x]
    xgb = [horizons[str(h)]["xgboost"]["mae"] for h in x]
    ax.plot(x, persistence, marker="o", linewidth=2, color="#B76E2B", label="Persistence baseline")
    ax.plot(x, xgb, marker="o", linewidth=2.3, color="#2E74B5", label="Selected XGBoost")
    ax.set_xticks(x, [f"{h} h" for h in x])
    ax.set_ylabel("PM₂.₅ MAE (µg/m³) — lower is better")
    ax.set_title("Held-out Oct–Dec 2025 forecast error")
    ax.grid(axis="y", color="#D9E0E7", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    path = FIG_DIR / "project_heldout_mae.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def add_cover(doc):
    add_para(doc, "PROJECT REPORT  |  PS-13", size=10, color=GREEN, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, before=105, after=14)
    add_para(doc, "Delhi AirCast", size=31, color=NAVY, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, after=5)
    add_para(doc, "AI-Powered Air Quality Forecasting", size=18, color=BLUE,
             align=WD_ALIGN_PARAGRAPH.CENTER, after=12)
    add_para(doc, "A data, feature-engineering, model-evaluation, and deployment report",
             size=11.5, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=28)
    add_para(doc, "Prepared for project review", size=11, color=INK,
             align=WD_ALIGN_PARAGRAPH.CENTER, after=5)
    add_para(doc, REPORT_DATE.strftime("%d %B %Y"), size=10, color=MUTED,
             align=WD_ALIGN_PARAGRAPH.CENTER, after=75)
    add_callout(
        doc,
        "EXECUTIVE TAKEAWAY",
        "The project built a leakage-aware, 39-station Delhi PM₂.₅ forecasting pipeline and selected XGBoost after comparing it with persistence, LSTM, TCN, a spatial GNN, and alternative feature sets. On the Oct–Dec 2025 holdout, XGBoost reduced MAE at every tested horizon, though the 24-hour gain was small. A separate two-station live PM₂.₅-index pilot is deployed; it is not the same model or AQI scale as the main CPCB network model.",
    )
    doc.add_page_break()


def main():
    roles = load_json("data/processed/dataset_roles.json")
    feature_manifest = load_json("data/processed/delhi_unified_forecast.manifest.json")
    panel_manifest = load_json("data/processed/cpcb_2024_25_hourly/manifest.json")
    final = load_json("data/runs/final_xgboost/final_evaluation.json")
    waqi = load_json("data/runs/waqi_live_index/evaluation.json")
    main_horizons = final["horizons"]
    create_pipeline_figure()
    mae_figure = create_mae_figure(main_horizons)

    doc = Document()
    style_document(doc)
    add_cover(doc)

    add_heading(doc, "1. Project goal and outcome")
    add_para(doc, "Problem statement: Citizens lack hyperlocal, short-term air-quality forecasts for planning outdoor activity, particularly in polluted cities. The proposed approach combines sensor and weather history, with a time-series or graph model and an interactive map.")
    add_para(doc, "Delhi AirCast implements a research prototype for Delhi: it forecasts station-level PM₂.₅ concentrations at 1, 3, 6, 12, and 24 hours; derives a clearly labelled PM₂.₅ AQI proxy for the main model; and provides a separate live feed and forecast pilot for two Delhi locations. The data pipeline and evaluation are offline and reproducible, while the live service has a narrower data/model scope.")
    add_callout(doc, "SCOPE NOTE", "The main selected forecaster is XGBoost, not LSTM or GNN. Neural and graph models were implemented and evaluated as challengers. Their results did not justify replacing XGBoost.", fill=PALE_GOLD)

    add_heading(doc, "2. End-to-end system")
    doc.add_picture(str(FIGURE), width=Inches(6.35))
    add_para(doc, "Figure 1. Main offline CPCB forecasting pipeline and the distinct WAQI live pilot.", size=9, color=MUTED, italic=True, after=8)
    add_para(doc, "The principal workflow is: acquire source-specific records; normalize station readings to an hourly panel; audit coverage and coordinates; create point-in-time features and direct future targets; train separate horizon models; compare on chronological holdouts; then package predictions for API/dashboard use. Keeping the live pilot separate avoids presenting a WAQI index as an official CPCB AQI.")

    add_heading(doc, "3. Data used and dataset dimensions")
    cpcb_rows, cpcb_cols = parquet_shape("data/processed/cpcb_2024_25_hourly/site_301.parquet")
    multi_rows, multi_cols = parquet_shape("data/processed/delhi_multistation_forecast.parquet")
    unified_rows, unified_cols = parquet_shape("data/processed/delhi_unified_forecast.parquet")
    firms_rows, firms_cols = parquet_shape("data/processed/firms/firms_daily_features.parquet")
    openaq_rows, openaq_cols = parquet_shape("data/processed/openaq/openaq_delhi_ncr_hourly.parquet")
    legacy_rows, legacy_cols = parquet_shape("data/processed/cpcb_2017_2023_aqi.parquet")
    airdelhi_rows, airdelhi_cols = parquet_shape("data/processed/airdelhi_grid_hourly.parquet")
    weather = load_json("data/raw/weather/delhi-central-hourly-2017-2025.json")
    cams = load_json("data/raw/cams/delhi-central-cams-global-hourly-2023-2025.json")
    add_para(doc, f"The primary target panel contains {panel_manifest['station_count']} Delhi stations and {panel_manifest['resource_count']} normalized source resources. Coverage is {feature_manifest['start_utc'][:10]} through {feature_manifest['end_utc'][:10]} UTC, with {feature_manifest['rows_written']:,} station-hour rows and 93.20% mean network hourly PM₂.₅ availability. Each station has an hourly Parquet panel; site_301 is {cpcb_rows:,} × {cpcb_cols}.")
    add_table(doc, ["Dataset / artifact", "Dimensions", "Role in this project"], [
        ["CPCB/OpenCity hourly station panel", f"39 × {cpcb_rows:,} rows per station; {feature_manifest['rows_written']:,} total station-hours", "Primary labels and station observations; 2024–25"],
        ["Multi-station engineered table", f"{multi_rows:,} × {multi_cols}", "Pollutant/weather lags, calendar, neighbour summaries, and horizon targets"],
        ["Unified training feature table", f"{unified_rows:,} × {unified_cols}", "Final XGBoost training table; all-source point-in-time features and five targets"],
        ["Open-Meteo historical weather", f"{len(weather['hourly']['time']):,} × {len(weather['hourly'])}", "Auxiliary observed weather history; the model joins only issue-time/past values"],
        ["Open-Meteo/CAMS air quality", f"{len(cams['hourly']['time']):,} × {len(cams['hourly'])}", "Auxiliary regional reanalysis; CAMS joins enforce a six-hour lag"],
        ["NASA FIRMS daily feature table", f"{firms_rows:,} × {firms_cols}", "Prior complete-day fire counts and fire-radiative-power summaries"],
    ], [3000, 2600, 3760], font_size=8.7)
    add_para(doc, f"The final Parquet schema has {unified_cols} columns: station/time identifiers, observed and engineered predictors, five direct PM₂.₅ target columns (1/3/6/12/24 h), and target timestamp metadata. Its manifest enumerates {len(feature_manifest['feature_columns'])} feature/identifier fields separately from the target list. The CPCB panel quality audit matched 38 of 39 station names to coordinates; site_106 (IGI Airport) is left unmatched rather than assigned a guessed location.")

    add_heading(doc, "3.1 Sources reviewed but not pooled into main-model training", 2)
    add_table(doc, ["Dataset", "Available processed shape", "Why not main training labels/features"], [
        ["OpenAQ Delhi/NCR", f"{openaq_rows:,} × {openaq_cols}", "Validation/reference only; source time coverage and station identities do not align cleanly with the CPCB training panel."],
        ["CPCB legacy AQI 2017–2023", f"{legacy_rows:,} × {legacy_cols}", "Older AQI-only data; not mixed into pollutant-concentration targets."],
        ["AirDelhi spatial grid", f"{airdelhi_rows:,} × {airdelhi_cols}", "2020–21 mobile/grid measurements; different period and measurement system."],
        ["Kaggle/reference collections", "10 curated source collections downloaded", "Comparison/provenance review; not part of the promoted feature contract."],
        ["CPCB live snapshot / WAQI", "Single snapshots or current readings", "Inference-time live inputs, not historical offline training labels."],
    ], [2550, 2300, 4510], font_size=8.7)
    add_para(doc, "OpenAQ and CPCB comparisons found 47,649 same-hour PM₂.₅ pairs across 35 stations, but about 80.7 µg/m³ MAE and 0.50 overall correlation. Because agreement varies substantially by station, OpenAQ was not treated as interchangeable CPCB ground truth.")
    add_callout(doc, "KAGGLE REPRODUCIBILITY BUNDLE", "The public Kaggle dataset contains the 39-station normalized CPCB panel, weather/CAMS source records, FIRMS daily features, manifests, and the final engineered feature table. It is a curated training bundle, not the full 5.6 GB exploration archive. Download: kaggle.com/datasets/priyanshuchawda/delhi-aircast-training-data")

    add_heading(doc, "4. Data quality and feature engineering")
    add_para(doc, "The hourly panel spans two full years across winter, monsoon, post-monsoon, and summer conditions. Original CPCB/OpenCity station resources are quarter-hourly; the pipeline averages readings to hourly timestamps, normalizes names and units, preserves missingness, and filters invalid sentinel readings. Coverage is measured before model fitting, and station identities are kept explicit.")
    add_table(doc, ["Feature family", "Engineering performed"], [
        ["Pollutant history", "PM₂.₅ lags at 1, 3, 6, 12, 24, 48, and 72 hours; rolling means and standard deviations over 3, 6, 12, 24, and 72 hours. Other available pollutants receive current and short/24-hour lag features."],
        ["Calendar", "Hour-of-day, weekday, and month encoded with sine/cosine pairs; weekend flag."],
        ["Spatial context", "Nearest coordinate-matched stations summarized at issue time using PM₂.₅ mean, maximum, spread, and availability count. The graph challenger uses six nearest neighbours."],
        ["Weather", "Temperature, humidity, rain, pressure, cloud cover, and wind. Wind is expressed as directional components; issue-time and historical lag features are used."],
        ["CAMS", "Regional PM₂.₅/PM₁₀ as-of values, 6/12/24-hour lags and rolling context; all values respect a six-hour publication lag."],
        ["Fire influence", "NASA FIRMS counts and fire radiative power by region for completed prior days, plus three- and seven-day summaries."],
        ["Targets / leakage controls", "Separate direct targets at +1/+3/+6/+12/+24 hours. Training/validation/test are chronological; predictor values are restricted to information available at issue time."],
    ], [2300, 7060], font_size=8.8)

    add_heading(doc, "5. Models and evaluation approach")
    add_para(doc, "The forecast is regression, so PM₂.₅ performance is reported with MAE and RMSE in µg/m³ and R². MAE is average absolute concentration error; RMSE penalizes large misses more heavily; R² measures variance explained relative to a constant-mean predictor. These are error metrics, not a single classification-style ‘accuracy’ percentage. A persistence forecast (future equals latest observed PM₂.₅) is the baseline.")
    add_table(doc, ["Candidate", "Implementation", "Decision"], [
        ["Persistence", "Last issue-time observation copied forward to each horizon.", "Reference baseline."],
        ["XGBoost", "Separate gradient-boosted tree regressor per horizon, trained on tabular engineered features.", "Selected for the main offline CPCB network forecast."],
        ["LSTM / TCN", "48-hour input sequence; train-only normalization, missingness masks and station embedding; direct six-hour target in the reported matched comparison.", "Evaluated as challengers; higher MAE than XGBoost on the same test cohort."],
        ["Spatial GNN", "39 stations as nodes; six-nearest-station graph; recent pollution/weather/time inputs and train-fitted normalization.", "Tested at all five horizons; did not beat XGBoost."],
        ["HistGradientBoosting / blend", "Alternative boosted-tree candidate and validation-selected blend in feature diagnostics.", "No validated reason to add a second production model."],
        ["Multi-pollutant AQI", "Additional six-hour XGBoost heads for PM₁₀, NO₂, CO, and O₃ combined with PM₂.₅ using CPCB sub-index logic.", "Exploratory AQI improvement; incomplete pollutant set and reused test period."],
    ], [1850, 4800, 2710], font_size=8.4)
    add_para(doc, "Evaluation protocol: training rows precede July 2025; July–September 2025 is the tuning/validation interval; October–December 2025 is the chronological test tail. Final models were refit on eligible data before the test tail and scored on about 80,000–82,000 station-hours per horizon. The model strategy diagnostic additionally uses a one-in-three deterministic training-row sample to limit CPU cost, while retaining the full validation/test rows.")

    add_heading(doc, "6. Main held-out results: selected XGBoost")
    add_para(doc, "The scorecard below comes from the saved final-model evaluation artifact, on the October–December 2025 holdout. Lower MAE/RMSE is better; higher R² is better.")
    doc.add_picture(str(mae_figure), width=Inches(6.15))
    add_para(doc, "Figure 2. Persistence and XGBoost PM₂.₅ MAE on the same reported final evaluation horizon splits.", size=9, color=MUTED, italic=True, after=7)

    mae_rows = []
    r2_rows = []
    for key, result in main_horizons.items():
        p = result["persistence"]
        xgb = result["xgboost"]
        gain = (p["mae"] - xgb["mae"]) / p["mae"] * 100
        mae_rows.append([f"{key} h", f"{result['test_rows']:,}", f"{p['mae']:.2f}", f"{xgb['mae']:.2f}", f"{gain:.1f}%"])
        r2_rows.append([f"{key} h", f"{p['rmse']:.2f}", f"{xgb['rmse']:.2f}", f"{p['r2']:.3f}", f"{xgb['r2']:.3f}"])
    add_table(doc, ["Horizon", "Test rows", "Persistence MAE", "XGBoost MAE", "MAE reduction"], mae_rows,
              [1300, 1800, 1900, 1900, 2460], font_size=9)
    add_para(doc, "MAE and test-row count are reported per forecast horizon; MAE reduction is relative to persistence on that horizon.", size=9, color=MUTED, italic=True, after=6)
    add_table(doc, ["Horizon", "Persistence RMSE", "XGBoost RMSE", "Persistence R²", "XGBoost R²"], r2_rows,
              [1300, 2000, 2000, 2030, 2030], font_size=9)
    add_para(doc, "XGBoost reduces the average PM₂.₅ error across the tested horizons. Its 24-hour MAE gain is modest (about 4.4%), so the longer horizon should be presented with appropriate uncertainty. MAE can look large in µg/m³ because the held-out period includes severe pollution episodes and a substantially harder seasonal distribution than the summer validation window.")

    add_heading(doc, "7. Candidate-model comparisons")
    add_heading(doc, "7.1 LSTM and TCN (same six-hour test cohort)", 2)
    add_table(doc, ["Model", "MAE (µg/m³)", "RMSE (µg/m³)", "Readout"], [
        ["Persistence", "66.812", "94.043", "Baseline"],
        ["XGBoost", "42.814", "63.032", "Lowest MAE in this comparison"],
        ["LSTM", "45.195", "69.235", "About 5.3% higher MAE than XGBoost"],
        ["TCN", "45.197", "68.380", "About 5.3% higher MAE than XGBoost"],
    ], [2100, 1900, 1900, 3460], font_size=9)
    add_para(doc, "The sequence models used a short CPU-prototyping run and are challengers, not deployed replacements. Their scores support the simpler tree model under the tested setup; they do not prove that every LSTM/TCN design is inferior.")

    add_heading(doc, "7.2 Spatial GNN (same-row comparison by horizon)", 2)
    gnn_rows = [
        ["1 h", "80,224", "22.879", "21.802", "18.887"],
        ["3 h", "79,664", "48.141", "37.979", "35.339"],
        ["6 h", "79,254", "70.650", "49.393", "44.246"],
        ["12 h", "78,780", "85.446", "58.956", "50.901"],
        ["24 h", "78,257", "58.672", "61.145", "56.117"],
    ]
    add_table(doc, ["Horizon", "Rows", "Persistence MAE", "GNN MAE", "XGBoost MAE"], gnn_rows,
              [1300, 1700, 2100, 1900, 2360], font_size=8.8)
    add_para(doc, "In this comparison, GNN beats persistence through 12 hours but loses to XGBoost at every horizon and also loses to persistence at 24 hours. AQI-category results derived from PM₂.₅ are proxies, not official composite-AQI classification results.")

    add_heading(doc, "7.3 Added-data feature ablation", 2)
    add_table(doc, ["Six-hour feature set / candidate", "Validation MAE", "Held-out MAE"], [
        ["CPCB station/time/neighbour features, XGBoost", "11.834", "47.930"],
        ["+ Open-Meteo weather", "12.117", "45.618"],
        ["+ weather and CAMS", "12.291", "44.547"],
        ["+ FIRMS (all sources)", "12.906", "44.283"],
        ["All sources, HistGradientBoosting", "—", "44.587"],
        ["Validation-selected tree blend", "—", "44.283"],
    ], [5100, 2100, 2160], font_size=8.8)
    add_para(doc, "Weather, CAMS, and FIRMS together reduce six-hour held-out MAE by 3.647 µg/m³ versus CPCB-only features in this sampled diagnostic. The gain is not uniform: auxiliary data slightly worsens the one-hour and 24-hour score in other tested comparisons. More data was therefore included only when temporally aligned, and feature usefulness was assessed horizon by horizon.")

    add_heading(doc, "8. AQI and separate live-pilot results")
    add_heading(doc, "8.1 Multi-pollutant CPCB AQI estimate", 2)
    add_para(doc, "Because PM₂.₅ alone is not the full CPCB AQI, a separate six-hour experiment forecast PM₂.₅, PM₁₀, NO₂, CO, and O₃ and applied CPCB sub-index calculations. On the same 81,074-row cohort, the five-pollutant subset scored AQI MAE 47.10 and category accuracy 62.10%; PM₂.₅-only proxy scored MAE 51.68 and category accuracy 57.77%; persistence scored MAE 75.97 and category accuracy 47.18%.")
    add_callout(doc, "AQI LIMITATION", "This is an incomplete five-pollutant estimate: SO₂ and NH₃ are omitted, the AQI uses the available pollutant subset, and the Oct–Dec 2025 test period had already been used in earlier candidate experiments. Treat it as an internal research result, not an untouched regulatory-grade benchmark.", fill=PALE_GOLD)

    add_heading(doc, "8.2 Live two-station PM₂.₅-index pilot", 2)
    add_para(doc, "A separate model path serves Pusa (site_107) and R.K. Puram (site_124). It uses a HistGradientBoostingRegressor trained from historical CPCB PM₂.₅ transformed to a legacy US-EPA-style PM₂.₅ index target. At inference, current WAQI PM₂.₅ index, local time, and station mapping are required. It predicts index points—not CPCB composite AQI or PM₂.₅ concentration.")
    live_rows = []
    for h, scores in waqi["results"].items():
        test = scores["test"]
        live_rows.append([f"{h} h", f"{test['rows']:,}", f"{test['persistence']['mae_index_points']:.2f}", f"{test['model']['mae_index_points']:.2f}", f"{test['model']['rmse_index_points']:.2f}"])
    add_table(doc, ["Horizon", "Test rows", "Persistence MAE", "Pilot MAE", "Pilot RMSE"], live_rows,
              [1300, 1800, 2050, 2050, 2260], font_size=8.9)
    add_para(doc, "The live pilot’s 2025 test MAE is lower than persistence through 12 hours; at 24 hours it is effectively tied (52.41 vs 52.49 index points). It is useful as a demonstration of a live-input/model flow, but only covers two stations and depends on a third-party feed.")

    add_heading(doc, "9. Application, deployment, and current operational boundary")
    add_table(doc, ["Layer", "Technology / location", "What is available"], [
        ["Dashboard", "Next.js / React, Vercel", "Public project UI: delhi-aircast-ps13.vercel.app"],
        ["Forecast API", "FastAPI, Render", "Health endpoint responds; serves the separate WAQI live-index pilot weights."],
        ["Main offline model", "Five XGBoost horizon artifacts", "Trained and evaluated locally; current Render health response reports no unified dataset or final XGBoost horizons configured."],
        ["Training/reproduction data", "Public Kaggle dataset", "Ready-to-train feature table and source panel with source notes."],
    ], [1800, 2400, 5260], font_size=8.8)
    p = add_para(doc, "Dashboard: ", size=10, after=3)
    add_hyperlink(p, "https://delhi-aircast-ps13.vercel.app", "https://delhi-aircast-ps13.vercel.app")
    p = add_para(doc, "API health: ", size=10, after=3)
    add_hyperlink(p, "https://delhi-aircast-api.onrender.com/health", "https://delhi-aircast-api.onrender.com/health")
    p = add_para(doc, "Training data: ", size=10, after=8)
    add_hyperlink(p, "https://www.kaggle.com/datasets/priyanshuchawda/delhi-aircast-training-data", "Kaggle training dataset")
    add_callout(doc, "IMPORTANT DEPLOYMENT STATUS", "At report generation, the API health endpoint is reachable and reports the AQI engine plus WAQI live-index horizons, but no main multistation/unified dataset or final XGBoost horizons are configured on Render. Therefore, describe the main CPCB network model as trained/evaluated research artifacts and the two-station WAQI path as the deployed live pilot. Do not claim all 39-station forecasts are currently served live.", fill=PALE_GOLD)

    add_heading(doc, "10. Limitations, risks, and next steps")
    add_table(doc, ["Limitation", "Why it matters / next improvement"], [
        ["Two-year primary target history", "Only two annual cycles; keep collecting CPCB data and re-evaluate across more winters and pollution regimes."],
        ["Seasonal distribution shift", "Summer validation errors are much lower than polluted Oct–Dec test errors; emphasize winter/high-pollution slices and underprediction."],
        ["Station-level, not continuous hyperlocal field", "39 monitors improve neighbourhood coverage but do not resolve every street; add validated interpolation/sensor fusion before claiming block-level precision."],
        ["Single central weather location", "Weather features are shared from central Delhi; station-specific weather grids or matched forecasts could improve local representation."],
        ["Incomplete composite AQI", "Current multi-pollutant six-hour estimate omits SO₂ and NH₃; complete pollutant heads, coverage rules and test evaluation."],
        ["Live pilot coverage and feed dependency", "Only Pusa and R.K. Puram; stale/missing WAQI and free-host cold starts require prominent freshness/fallback states."],
        ["No calibrated prediction intervals", "Add horizon-specific quantiles/conformal intervals and report empirical coverage and width."],
    ], [2900, 6560], font_size=8.7)
    add_para(doc, "Recommended academic conclusion: the project demonstrates a complete data-to-model forecasting workflow and a measured XGBoost improvement over persistence on a chronological holdout. It is a strong prototype, not a claim of best-in-class or operational health guidance. The most valuable next work is multi-season re-evaluation, high-pollution calibration, full CPCB AQI pollutant coverage, and making the offline 39-station serving artifacts available to the deployed API if that product scope is required.")

    add_heading(doc, "11. Reproducibility")
    add_para(doc, "The code repository contains the acquisition, normalization, feature engineering, model training, and evaluation scripts. Download the public Kaggle bundle, place its source files at the project’s documented data paths, and use the following commands from a clean checkout:")
    for command in (
        "uv sync --group dev",
        "uv run python scripts/analyze_cpcb_panel.py",
        "uv run python scripts/build_multistation_dataset.py",
        "uv run python scripts/build_unified_dataset.py",
        "uv run python scripts/train_final_models.py --n-jobs 4",
        "uv run python scripts/train_sequence_model.py --model lstm --horizon 6 --sequence-length 48 --stride 12 --epochs 3",
        "uv run python scripts/train_spatial_gnn.py --horizon 6 --epochs 20 --lr 0.001 --patience 5",
    ):
        p = add_para(doc, command, size=9.2, color=NAVY, after=3, keep=True)
        p.paragraph_format.left_indent = Inches(0.25)

    add_heading(doc, "12. Short presentation summary")
    add_callout(doc, "IN ONE MINUTE", "Delhi AirCast uses 2024–25 CPCB observations from 39 Delhi monitoring stations (684,216 hourly station records). We engineered pollutant history, rolling statistics, calendar cycles, nearby-station summaries, weather, six-hour-lagged CAMS, and prior-day NASA fire features into a 684,216 × 151 Parquet table. Separate XGBoost models predict PM₂.₅ one to 24 hours ahead. On an Oct–Dec 2025 chronological test, they beat persistence at each horizon; six-hour MAE was 43.89 versus 69.95 µg/m³. LSTM, TCN, and GNN challengers were tested but did not beat XGBoost under the evaluated splits. A public Kaggle bundle shares training data. A separate two-station WAQI index pilot is deployed, while the main 39-station XGBoost artifacts are not currently configured on the hosted API.")

    add_heading(doc, "Sources and project artifacts")
    sources = [
        ("Problem statement", "problem.md"),
        ("Measured experiment ledger", "EXPERIMENTS.md"),
        ("Data-source inventory and roles", "DATA_SOURCES.md; data/processed/dataset_roles.json"),
        ("Primary data", "OpenCity Delhi Hourly Air Quality Reports: https://data.opencity.in/dataset/delhi-hourly-air-quality-reports"),
        ("Weather", "Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api"),
        ("CAMS", "Open-Meteo Air Quality API: https://open-meteo.com/en/docs/air-quality-api"),
        ("Fire observations", "NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/active_fire/"),
        ("AQI method", "CPCB National Air Quality Index: https://airquality.cpcb.gov.in/ccr_docs/How_AQI_Calculated.pdf"),
        ("Dataset publication", "Kaggle: https://www.kaggle.com/datasets/priyanshuchawda/delhi-aircast-training-data"),
    ]
    for label, value in sources:
        add_mixed_para(doc, [(f"{label}: ", {"bold": True, "color": NAVY}), (value, {})], after=3, size=9.5)
    add_para(doc, "Metric source: saved local evaluation artifacts under data/runs/final_xgboost/, data/runs/spatial_gnn/, and data/runs/waqi_live_index/. The experiment ledger documents the exact matched-cohort comparisons and their limitations.", size=9.2, color=MUTED, italic=True, before=6)

    doc.core_properties.title = "Delhi AirCast PS-13 Project Report"
    doc.core_properties.subject = "Data, feature engineering, model evaluation, and deployment"
    doc.core_properties.author = "Delhi AirCast Project"
    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
