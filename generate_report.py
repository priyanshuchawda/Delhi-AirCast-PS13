#!/usr/bin/env python3
"""Generate the PS-13 Delhi AirCast data-acquisition report (DOCX)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


OUT_DIR = Path(__file__).resolve().parent
OUT_DOCX = OUT_DIR / "Delhi_AirCast_PS13_Data_Acquisition_Report.docx"
FIG_DIR = OUT_DIR / "figures"

NAVY = RGBColor(0x1B, 0x3A, 0x5F)
ACCENT = RGBColor(0x2C, 0x5F, 0x8A)
DARK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEADER_BG = "1B3A5F"
ALT_BG = "EEF3F8"
OK_BG = "E7F6EC"
WARN_BG = "FFF4E5"
NO_BG = "FDECEC"

TODAY = date(2026, 9, 7)


def set_run(run, *, size=11, bold=False, italic=False, color=DARK, font="Times New Roman"):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color


def add_text(p, text, **kwargs):
    run = p.add_run(text)
    set_run(run, **kwargs)
    return run


def shade_cell(cell, hex_color: str):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def set_cell_borders(cell, color="CCCCCC"):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        tcBorders.append(el)
    tcPr.append(tcBorders)


def set_cell_text(cell, text, *, bold=False, size=9.5, color=DARK, align="left", font="Times New Roman"):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
    }[align]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.08
    add_text(p, text, size=size, bold=bold, color=color, font=font)
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.font.name = font
            run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def make_table(doc, headers, rows, col_widths=None, caption=None):
    if caption:
        cap = doc.add_paragraph()
        cap.paragraph_format.space_before = Pt(10)
        cap.paragraph_format.space_after = Pt(4)
        add_text(cap, caption, size=10, italic=True, color=ACCENT)

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        set_cell_text(cell, h, bold=True, size=9, color=WHITE, align="center")
        shade_cell(cell, HEADER_BG)
        set_cell_borders(cell, "1B3A5F")

    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = table.rows[r_i + 1].cells[c_i]
            text = "" if val is None else str(val)
            align = "left" if c_i == 0 else "left"
            set_cell_text(cell, text, size=9, color=DARK, align=align)
            shade_cell(cell, ALT_BG if r_i % 2 else "FFFFFF")
            set_cell_borders(cell, "D0D7DE")
            # colour-code first-column verdicts
            low = text.strip().lower()
            if low.startswith("use") or low == "primary" or low.startswith("verified"):
                shade_cell(cell, OK_BG)
            elif low.startswith("backup") or low.startswith("auxiliary") or low.startswith("optional"):
                shade_cell(cell, WARN_BG)
            elif low.startswith("reject") or low.startswith("do not") or low == "skip":
                shade_cell(cell, NO_BG)

    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Cm(w)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = NAVY
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    p.paragraph_format.space_before = Pt(16 if level == 1 else 12)
    p.paragraph_format.space_after = Pt(6)
    return p


def para(doc, text, *, first_line=True, space_after=8, italic=False, bold=False, size=11, align="justify"):
    p = doc.add_paragraph()
    p.alignment = {
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
    }[align]
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.18
    if first_line and align == "justify":
        pf.first_line_indent = Cm(0.75)
    add_text(p, text, size=size, italic=italic, bold=bold)
    return p


def bullet(doc, text, *, bold_lead=None, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(1.0 + 0.5 * level)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.12
    if bold_lead:
        add_text(p, bold_lead, size=11, bold=True)
        add_text(p, text, size=11)
    else:
        add_text(p, text, size=11)
    return p


def numbered(doc, text, *, bold_lead=None):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.space_after = Pt(3)
    if bold_lead:
        add_text(p, bold_lead, size=11, bold=True)
        add_text(p, text, size=11)
    else:
        add_text(p, text, size=11)
    return p


def quote_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.2)
    p.paragraph_format.right_indent = Cm(1.0)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(8)
    add_text(p, text, size=11, italic=True, color=ACCENT)
    return p


def add_figure(doc, filename: str, caption: str, width_cm: float = 16.0):
    path = FIG_DIR / filename
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    if path.exists():
        run = p.add_run()
        run.add_picture(str(path), width=Cm(width_cm))
    else:
        add_text(p, f"[Missing figure: {filename}]", size=10, italic=True, color=MUTED)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    add_text(cap, caption, size=10, italic=True, color=ACCENT)
    return cap


def page_break(doc):
    doc.add_page_break()


def add_page_number_field(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run(run, size=9, color=MUTED)


def add_num_pages_field(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " NUMPAGES "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run(run, size=9, color=MUTED)


def setup_header_footer(doc):
    section = doc.sections[0]
    section.different_first_page_header_footer = True

    # blank first-page header/footer
    section.first_page_header.paragraphs[0].text = ""
    section.first_page_footer.paragraphs[0].text = ""

    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_text(hp, "PS-13  ·  Delhi AirCast  ·  Data Acquisition Report", size=9, italic=True, color=MUTED)

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(fp, "Page ", size=9, color=MUTED)
    add_page_number_field(fp)
    add_text(fp, " of ", size=9, color=MUTED)
    add_num_pages_field(fp)


def setup_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.font.color.rgb = DARK
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    pf = normal.paragraph_format
    pf.line_spacing = 1.18
    pf.space_after = Pt(8)

    for i in range(1, 4):
        h = styles[f"Heading {i}"]
        h.font.name = "Times New Roman"
        h.font.color.rgb = NAVY
        h.font.bold = True
        h._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        if i == 1:
            h.font.size = Pt(16)
        elif i == 2:
            h.font.size = Pt(13)
        else:
            h.font.size = Pt(12)


def horizontal_line(paragraph):
    p = paragraph._p
    pPr = p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "12")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "1B3A5F")
    pBdr.append(bottom)
    pPr.append(pBdr)


def cover_page(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(p, "PROBLEM STATEMENT  PS-13", size=12, bold=True, color=ACCENT)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    add_text(p, "AI-Powered Air Quality Forecasting", size=14, bold=True, color=NAVY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    add_text(p, "Delhi AirCast", size=28, bold=True, color=NAVY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(
        p,
        "Hyperlocal Short-Term PM2.5 and AQI Forecasting\nfor Individual Monitoring Stations in Delhi",
        size=14,
        italic=True,
        color=ACCENT,
    )

    line = doc.add_paragraph()
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    horizontal_line(line)
    add_text(line, " ", size=6)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    add_text(
        p,
        "Data Sources, Acquisition Architecture\nand System Design Report",
        size=16,
        bold=True,
        color=DARK,
    )

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    add_text(
        p,
        "Free Indian government and weather data, checked and ready to use,\n"
        "for a neighbourhood air-quality forecast in Delhi.",
        size=11,
        italic=True,
        color=MUTED,
    )

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(22)
    add_text(p, "Undergraduate Technical Project Report", size=12, bold=True, color=NAVY)

    meta = [
        ("City / study region", "Delhi, National Capital Territory of India"),
        ("Primary target", "Hourly PM2.5 (µg/m³), converted to Indian National AQI"),
        ("Forecast horizon", "1–24 hours ahead, neighbourhood / station granularity"),
        ("Verification date", "7 September 2026"),
        ("Status", "Verified system design  ·  7 September 2026"),
    ]
    table = doc.add_table(rows=len(meta), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (k, v) in enumerate(meta):
        set_cell_text(table.rows[i].cells[0], k, bold=True, size=10, color=NAVY)
        set_cell_text(table.rows[i].cells[1], v, size=10)
        shade_cell(table.rows[i].cells[0], "EEF3F8")
        set_cell_borders(table.rows[i].cells[0], "D0D7DE")
        set_cell_borders(table.rows[i].cells[1], "D0D7DE")
        table.rows[i].cells[0].width = Cm(5.5)
        table.rows[i].cells[1].width = Cm(11.0)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(p, "Submitted in partial fulfilment of the project requirement", size=11, italic=True, color=MUTED)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    add_text(p, "Project team", size=12, bold=True, color=NAVY)

    team = [
        ("1", "Priyanshu Chawda"),
        ("2", "Aryan Babel"),
        ("3", "Shruti Agrawal"),
        ("4", "Aditya Gayal"),
    ]
    t_team = doc.add_table(rows=len(team), cols=2)
    t_team.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (n, name) in enumerate(team):
        set_cell_text(t_team.rows[i].cells[0], n, bold=True, size=12, color=NAVY, align="center")
        set_cell_text(t_team.rows[i].cells[1], name, size=12, color=DARK)
        shade_cell(t_team.rows[i].cells[0], "EEF3F8")
        set_cell_borders(t_team.rows[i].cells[0], "D0D7DE")
        set_cell_borders(t_team.rows[i].cells[1], "D0D7DE")
        t_team.rows[i].cells[0].width = Cm(1.6)
        t_team.rows[i].cells[1].width = Cm(10.0)

    fields = [
        ("Programme / year", "________________________________"),
        ("Institution", "________________________________"),
        ("Project guide", "________________________________"),
        ("Date", TODAY.strftime("%d %B %Y")),
    ]
    t2 = doc.add_table(rows=len(fields), cols=2)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (k, v) in enumerate(fields):
        set_cell_text(t2.rows[i].cells[0], k, bold=True, size=11, color=DARK)
        set_cell_text(t2.rows[i].cells[1], v, size=11)
        t2.rows[i].cells[0].width = Cm(5.0)
        t2.rows[i].cells[1].width = Cm(11.5)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(24)
    add_text(
        p,
        "All sources used in this report are free for academic use.\n"
        "Live endpoints were probed on 7 September 2026 before the design was frozen.",
        size=9,
        italic=True,
        color=MUTED,
    )


def toc_page(doc):
    page_break(doc)
    heading(doc, "Contents", 1)
    items = [
        ("1.", "Introduction"),
        ("2.", "Problem Statement and Objectives"),
        ("3.", "Scope, Assumptions and Design Principles"),
        ("4.", "What We Are Building"),
        ("5.", "Verified Data Landscape"),
        ("6.", "Layer A — Primary Historical Air Quality Data"),
        ("7.", "Layer B — Official Live CPCB Snapshot"),
        ("8.", "Layer C — OpenAQ Archive and Live Backup"),
        ("9.", "Layer D — Weather: Open-Meteo"),
        ("10.", "Layer E — Auxiliary Free Sources"),
        ("11.", "Other Public Datasets"),
        ("12.", "Recommended Multi-Source Architecture"),
        ("13.", "Target Variable and Indian AQI Methodology"),
        ("14.", "Dataset Schema and Feature Engineering"),
        ("15.", "Data Collection, Storage and Update Cadence"),
        ("16.", "Quality Control and Missing-Value Policy"),
        ("17.", "Modelling Strategy"),
        ("18.", "Evaluation, Drift and Continuous Learning"),
        ("19.", "Dashboard Specification"),
        ("20.", "Free Accounts, Keys and Licensing"),
        ("21.", "Keeping the System Reliable"),
        ("22.", "Implementation Roadmap"),
        ("23.", "Future Scope and Growth"),
        ("24.", "Conclusion"),
        ("25.", "References"),
        ("Appendix A.", "Delhi Monitoring Stations in the OpenCity Dump"),
        ("Appendix B.", "Live Verification Log (7 September 2026)"),
        ("Appendix C.", "CPCB National AQI Breakpoints"),
        ("Appendix D.", "Worked Forecast Example"),
        ("Appendix E.", "List of Figures"),
    ]
    for num, title in items:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(1)
        tab = p.paragraph_format.tab_stops
        tab.add_tab_stop(Cm(2.2), WD_TAB_ALIGNMENT.LEFT)
        add_text(p, f"{num}\t{title}", size=11)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    add_text(
        p,
        "Note.  This document is the data, architecture and modelling plan for Delhi AirCast. "
        "We build the project from this plan.",
        size=10,
        italic=True,
        color=MUTED,
    )


def build_body(doc):
    page_break(doc)
    heading(doc, "1.  Introduction", 1)
    para(
        doc,
        "People in Delhi need air information they can use: their own neighbourhood, and the next few hours. "
        "That helps someone decide whether to run at 6 a.m., whether a school playground stays open at 4 p.m., "
        "or how the next six hours look in their area. Problem Statement PS-13 asks for an AI system that "
        "forecasts air quality at neighbourhood scale using station readings, weather, and links between nearby "
        "monitoring stations.",
    )
    para(
        doc,
        "We start with good data, then we add the AI. This report shows the free CPCB and weather sources "
        "we checked on 7 September 2026, and the clean hourly table we build from them for Delhi.",
    )
    para(
        doc,
        "The system is named Delhi AirCast. It predicts PM2.5 and the Indian "
        "National Air Quality Index 1–24 hours ahead for individual monitoring locations, using historical "
        "pollution, meteorology, time features and, in a later phase, neighbouring-station information.",
    )

    heading(doc, "2.  Problem Statement and Objectives", 1)
    heading(doc, "2.1  Formal problem statement", 2)
    para(
        doc,
        "PS-13 — AI-Powered Air Quality Forecasting. Citizens lack hyperlocal, short-term air quality "
        "forecasts to plan outdoor activity, especially in polluted cities. The suggested solution is a "
        "time-series / graph neural network model that combines sensor and weather data to forecast AQI at "
        "neighbourhood granularity. Suggested technology includes LSTM / Graph Neural Network, CPCB sensor "
        "data, a weather API, and an interactive map dashboard.",
    )
    heading(doc, "2.2  Objectives of this report", 2)
    numbered(doc, "Identify every usable free air-quality and weather source relevant to Delhi.")
    numbered(doc, "Distinguish ground-station measurements from satellite / chemical-transport model fields.")
    numbered(doc, "Check which free endpoints respond quickly enough for a college project.")
    numbered(doc, "Set a clear job for each source: training data, live updates, weather, and extra background.")
    numbered(doc, "Define the training table, live update cadence, AQI conversion, modelling stages and dashboard.")
    numbered(doc, "Lay out the model path: simple baseline, then XGBoost, then LSTM, then a station graph.")

    heading(doc, "3.  Scope, Assumptions and Design Principles", 1)
    heading(doc, "3.1  Scope", 2)
    bullet(doc, " Delhi (NCT) for Phase 1. The same method can later cover Pune and other cities.", bold_lead="Geography.")
    bullet(doc, " Eight to ten CPCB/DPCC/IMD/IITM stations spanning east, west, south, north and the airport corridor.", bold_lead="Stations.")
    bullet(doc, " Forecasts 1, 6, 12 and 24 hours ahead.", bold_lead="Horizon.")
    bullet(doc, " Hourly. 15-minute CPCB files are averaged to the hour for modelling.", bold_lead="Resolution.")
    bullet(doc, " The model predicts PM2.5. The app converts that number to Indian AQI.", bold_lead="Target.")
    bullet(doc, " Free academic sources: OpenCity, data.gov.in, OpenAQ and Open-Meteo.", bold_lead="Cost.")

    heading(doc, "3.2  Design principles", 2)
    bullet(doc, " We train on CPCB station readings and show live CPCB updates on the dashboard.", bold_lead="Real station data.")
    bullet(doc, " Weather comes from Open-Meteo, which is free and works in under a second.", bold_lead="Fast weather.")
    bullet(doc, " We start with a simple baseline, then XGBoost, then LSTM, then a station graph. Each step has to improve the last one.", bold_lead="Step by step.")
    bullet(doc, " The model learns patterns from past pollution, weather, time of day and nearby stations.", bold_lead="Clear method.")

    heading(doc, "4.  What We Are Building", 1)
    para(
        doc,
        "Given current and recent pollution plus weather at one Delhi station, forecast PM2.5 for the next "
        "few hours, convert that forecast to Indian AQI, and tell a citizen whether outdoor air is expected "
        "to improve or deteriorate. A concrete example:",
    )
    quote_block(
        doc,
        "At 10:00 at Anand Vihar: PM2.5 = 145 µg/m³, PM10 = 230, temperature = 27 °C, humidity = 72 %, "
        "wind = 4 km/h, no rain. The model emits a six-hour PM2.5 path (for example 151, 158, 163, 157, "
        "149, 142), an AQI near 190 (Poor), and a sentence: “Air quality is expected to deteriorate over "
        "the next three hours.”",
    )
    para(
        doc,
        "That is already a legitimate AI forecasting system. Neighbourhood granularity comes from modelling "
        "stations separately (and, later, as a graph). Short-term comes from the 1–24 h horizon. The map "
        "dashboard is the public interface. Every week we add new data and keep the better model.",
    )

    heading(doc, "5.  Verified Data Landscape", 1)
    para(
        doc,
        "We use three kinds of free data, and each has a clear job.",
        first_line=True,
    )
    make_table(
        doc,
        ["Class", "What it actually is", "Use in Delhi AirCast"],
        [
            [
                "Ground station (CPCB / DPCC / IMD / IITM)",
                "A physical analyser at a named site, typically 15-min or hourly PM2.5, PM10, NO2, SO2, CO, O3, plus on-site meteorology.",
                "PRIMARY label and live current conditions",
            ],
            [
                "Meteorological reanalysis / forecast",
                "Gridded weather (ERA5 / ECMWF IFS) interpolated to each station.",
                "PRIMARY weather features and future-hour weather",
            ],
            [
                "Chemical transport / CAMS model",
                "Satellite-assimilated regional air quality at about 45 km, smooth and complete.",
                "EXTRA background feature and a useful comparison",
            ],
        ],
        caption="Table 1.  Three classes of free data, and the role of each in Delhi AirCast.",
    )
    para(
        doc,
        "Table 2 summarises every source that was either downloaded or called on 7 September 2026. "
        "Latency numbers are wall-clock times from this workstation on 7 September 2026.",
        first_line=False,
    )
    make_table(
        doc,
        ["Verdict", "Source", "Free?", "Key?", "What we got", "Latency"],
        [
            ["USE — primary train", "OpenCity CPCB dumps", "Yes, public domain", "No", "Anand Vihar 15-min, 2024-01-01 to 2025-12-31, 70,177 rows, full pollutants + station meteo", "3.4 s / 11.8 MB"],
            ["USE — live official", "data.gov.in CPCB API", "Yes (NDSAP)", "Free key", "Delhi stations updating 07-09-2026 21:00 IST (Narela, Rohini, Anand Vihar, NSIT Dwarka, …)", "~3 s"],
            ["USE — weather", "Open-Meteo forecast + archive", "Yes", "No", "Delhi hourly temp/humidity/wind/rain/pressure; archive week of 2024-01-01 returned 168 hours", "0.5–0.7 s"],
            ["USE — hist/live backup", "OpenAQ v3", "Yes", "Free key", "Free key needed. CPCB Anand Vihar is location 235, reporting since 2016", "ready with key"],
            ["EXTRA feature", "Open-Meteo Air Quality (CAMS)", "Yes", "No", "Regional background PM2.5 for Delhi, 21:30 IST = 103.1 µg/m³", "0.55 s"],
            ["EXTRA live", "WAQI / aqicn", "Yes", "Free token", "Third live feed once we have a personal token", "ready"],
            ["EXTRA history", "OpenAQ + Vonter CPCB archives", "Yes", "Free key / none", "More years and more stations when we want a longer record", "ready"],
        ],
        caption="Table 2.  Source scorecard after live verification on 7 September 2026.",
    )

    heading(doc, "6.  Layer A — Primary Historical Air Quality Data", 1)
    heading(doc, "6.1  OpenCity CPCB / DPCC station files (recommended core)", 2)
    para(
        doc,
        "The strongest training source found is the OpenCity dataset “Delhi Hourly Air Quality Reports”. "
        "It republishes CPCB Continuous Ambient Air Quality Monitoring Station (CAAQMS) files for 39 Delhi "
        "sites. The 2024–2025 series is 15-minute resolution with pollutant concentrations and on-site "
        "meteorology. Licence is listed as public domain. The original source cited is the CPCB data "
        "repository. We use that official Indian monitoring data through a fast public copy that is easy "
        "to download.",
    )
    para(
        doc,
        "Direct package API: https://data.opencity.in/api/3/action/package_show?id=delhi-hourly-air-quality-reports "
        "(HTTP 200 in 0.19 s; 78 resources). The Anand Vihar 2024–25 file was downloaded in full and profiled.",
        first_line=False,
    )
    make_table(
        doc,
        ["Field", "Observed value (Anand Vihar, site_301)"],
        [
            ["File", "del-anand-vihar-dpcc-2024-25.csv (11.8 MB)"],
            ["Rows", "70,177 (15-minute)"],
            ["Time span", "2024-01-01 00:00 UTC → 2025-12-31 23:45 UTC"],
            ["Pollutants", "PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, BTX"],
            ["Station meteorology", "AT (°C), RH (%), WS (m/s), WD (deg), RF (mm), BP, SR"],
            ["PM2.5 missing", "10,626 rows (15.1 %) — usual station gaps; we mark them and keep going"],
            ["PM2.5 range", "1.0 – 998 µg/m³ (998 is a known sensor spike; we clean it in QC)"],
            ["Licence", "Public Domain (OpenCity metadata)"],
        ],
        caption="Table 3.  Profile of the Anand Vihar 2024–2025 OpenCity extract.",
    )
    para(
        doc,
        "Ten stations hourly-aggregated over two years give on the order of 10 × 24 × 365 × 2 ≈ 175,000 "
        "station-hours. That is a complete training set. Two full years already include winter, monsoon, "
        "post-monsoon and Diwali. We use the 2024–2025 15-minute files because they have the full pollutant "
        "columns and a timestamp on every row.",
        first_line=False,
    )
    heading(doc, "6.2  How we get CPCB history quickly", 2)
    para(
        doc,
        "The official CAAQMS files come from CPCB. OpenCity, OpenAQ and the Vonter parquet releases already "
        "publish those same station readings in an easy, citable form. CPCB is the origin; OpenCity and OpenAQ "
        "are the fast access path we use.",
    )

    heading(doc, "7.  Layer B — Official Live CPCB Snapshot", 1)
    para(
        doc,
        "India’s Open Government Data platform publishes “Real Time Air Quality Index from various locations” "
        "(resource ID 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69), contributed by CPCB under the Ministry of "
        "Environment, Forest and Climate Change. A sample-key call with city = Delhi and pollutant = PM2.5 "
        "returned HTTP 200 and stations whose last_update was 07-09-2026 21:00:00 — the same evening this "
        "report was written. Example stations in that payload: Narela, Rohini, Sri Aurobindo Marg, Najafgarh, "
        "Talkatora Garden, NSIT Dwarka, Anand Vihar, Vivek Vihar.",
    )
    para(
        doc,
        "This endpoint is the official live snapshot for the dashboard. A personal data.gov.in key lets us "
        "pull every Delhi station and every pollutant in one job. Pollutant identifiers observed in community "
        "documentation: PM2.5, PM10, NO2, NH3, SO2, CO, OZONE.",
        first_line=False,
    )
    para(
        doc,
        "Field names in the JSON are min_value / avg_value / max_value. We train on OpenCity / OpenAQ "
        "concentrations in µg/m³. On the live dashboard we show the official CPCB number clearly, and we "
        "check it against the same hour in OpenCity so the units stay clear.",
    )
    quote_block(
        doc,
        "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
        "?api-key=YOUR_KEY&format=json&filters[city]=Delhi&limit=1000",
    )

    heading(doc, "8.  Layer C — OpenAQ Archive and Live Backup", 1)
    para(
        doc,
        "OpenAQ aggregates reference-grade monitors worldwide and exposes them through a documented REST API "
        "(v3). India CPCB stations, including Anand Vihar, New Delhi – DPCC, are present (OpenAQ location 235; "
        "reporting since 2016; parameters include PM2.5, PM10, NO2, SO2, CO, O3 and some on-site meteorology). "
        "A free key from explore.openaq.org unlocks the full API.",
    )
    para(
        doc,
        "OpenAQ gives us extra years before 2024, a second live path, and concentrations in µg/m³. It is "
        "the second free key in the stack, after data.gov.in.",
        first_line=False,
    )

    heading(doc, "9.  Layer D — Weather: Open-Meteo", 1)
    para(
        doc,
        "Weather is the easy part, and it should stay easy. Open-Meteo requires no API key for ordinary "
        "academic use. Two endpoints were verified:",
    )
    make_table(
        doc,
        ["Endpoint", "Query", "Result"],
        [
            [
                "api.open-meteo.com/v1/forecast",
                "Delhi 28.6139, 77.2090; hourly temp, RH, rain, pressure, wind, cloud; past 2 days + 1–2 day forecast; Asia/Kolkata",
                "HTTP 200 in 0.53 s; 72-hour series, first 2026-09-05 00:00, last 2026-09-07 23:00",
            ],
            [
                "archive-api.open-meteo.com/v1/archive",
                "Same point, 2024-01-01 to 2024-01-07",
                "HTTP 200 in 0.74 s; 168 hours; temperature at 00:00 IST = 7.0 °C",
            ],
        ],
        caption="Table 4.  Open-Meteo weather probes (7 September 2026).",
    )
    para(
        doc,
        "OpenCity files already include station weather, and Open-Meteo adds a complete history plus the "
        "next day’s forecast. That gives the model tomorrow’s wind, rain and temperature while it predicts "
        "the next hours of PM2.5 from recent readings.",
        first_line=False,
    )

    heading(doc, "10.  Layer E — Auxiliary Free Sources", 1)
    heading(doc, "10.1  Open-Meteo Air Quality (CAMS) — extra regional background", 2)
    para(
        doc,
        "Open-Meteo also serves Copernicus Atmosphere Monitoring Service (CAMS) global composition. The "
        "global domain is about 0.4° (~45 km), updated twice daily, with a multi-day forecast. A live call "
        "for Delhi returned PM2.5 = 103.1 µg/m³, PM10 = 113.9, US AQI = 159 at 2026-09-07 21:30. A 2024-01-01 "
        "week of history returned 168 hourly PM2.5 values (example 146.7, 130.4, 117.8). We use this as extra "
        "regional background, as a second comparison, and as a helpful fill when a station is briefly down. "
        "The station forecast still comes from CPCB readings.",
    )
    heading(doc, "10.2  WAQI / World Air Quality Index", 2)
    para(
        doc,
        "WAQI republishes many government stations and offers a free non-commercial token. A personal token "
        "gives us an extra live feed for the dashboard.",
    )
    heading(doc, "10.3  Daily CPCB AQI bulletins (UrbanEmissions)", 2)
    para(
        doc,
        "UrbanEmissions.info maintains processed CPCB daily AQI bulletins for Indian cities (2015–2024). "
        "We use these city-day numbers for the exploration chapter: winter vs monsoon, Diwali spikes, and "
        "year-to-year drift of city AQI. GitHub: urbanemissionsinfo/AQI_bulletins.",
    )
    heading(doc, "10.4  Vonter india-cpcb-aqi parquet", 2)
    para(
        doc,
        "The GitHub project Vonter/india-cpcb-aqi publishes 15-minute national CAAQMS extracts and hourly "
        "AQI parquet files (latest release tag 2025). If OpenCity is missing a Delhi station-year, this is "
        "another free CPCB archive we can add.",
    )
    heading(doc, "10.5  Vayuayan", 2)
    para(
        doc,
        "The vayuayan Python package wraps CPCB historical and live clients. We can use it later if we want "
        "another way to pull CPCB files.",
    )

    heading(doc, "11.  Other Public Datasets", 1)
    para(
        doc,
        "Many public Delhi files exist on Hugging Face, Kaggle and Zenodo. They are useful for extra reading "
        "and for later experiments. For training and for the live app we stay with OpenCity CPCB station "
        "files, the official data.gov.in feed, OpenAQ, and Open-Meteo, because we have already opened those "
        "files and confirmed the hours, stations and units.",
    )

    heading(doc, "12.  Recommended Multi-Source Architecture", 1)
    para(
        doc,
        "Delhi AirCast uses several free sources, each with a clear job: station readings for training, "
        "weather for features, the official API for live “now”, and extra feeds as backup.",
        first_line=True,
    )
    make_table(
        doc,
        ["Layer", "Source", "Cadence", "Role"],
        [
            ["A  Historical labels", "OpenCity CPCB 2024–2025 (8–10 Delhi stations)", "One-time bulk + optional yearly refresh", "Train / validate PM2.5"],
            ["A2 Extra years", "OpenAQ CPCB measurements; Vonter parquet if needed", "One-time back-fill", "2018–2023 optional"],
            ["B  Live official", "data.gov.in CPCB resource 3b01bcb8-…", "Every hour", "Dashboard “now”"],
            ["C  Live / hist backup", "OpenAQ v3 locations and measurements", "Every hour / on B failure", "Backup now + archive"],
            ["D  Weather hist+fcst", "Open-Meteo archive + forecast", "Hourly; 48 h forecast", "Features now and future"],
            ["E1 Regional AQ model", "Open-Meteo CAMS air-quality", "Hourly", "Auxiliary feature / baseline"],
            ["E2 Optional live", "WAQI personal token", "Hourly if B and C fail", "Tertiary nowcast"],
            ["E3 City climate EDA", "UrbanEmissions daily bulletins", "Static", "Report graphics only"],
        ],
        caption="Table 5.  Frozen multi-layer free-data architecture.",
    )
    para(
        doc,
        "Data flow in one sentence. OpenCity (and optionally OpenAQ) plus Open-Meteo history are cleaned and "
        "merged into a station-hour table; models are trained offline; every hour the live CPCB snapshot and "
        "Open-Meteo forecast are written to a database; the forecast engine emits PM2.5 paths; an AQI engine "
        "converts them; the web dashboard draws the map, charts and alerts.",
        first_line=False,
    )
    quote_block(
        doc,
        "CPCB/OpenAQ  +  Open-Meteo  →  clean & merge  →  train (XGBoost / LSTM [ / GNN ])  →  "
        "PM2.5 forecast  →  CPCB AQI  →  React (or similar) map dashboard.  "
        "New measurements every hour.  Retrain every week only if validation improves.",
    )
    add_figure(
        doc,
        "fig01_architecture.png",
        "Figure 1.  End-to-end Delhi AirCast stack. Green boxes are Phase-1 primary sources; amber boxes are backups or auxiliary features.",
        16.2,
    )
    add_figure(
        doc,
        "fig06_source_layers.png",
        "Figure 2.  Training labels, weather features and live updates each have a clear job.",
        15.4,
    )
    heading(doc, "12.1  What already works today", 2)
    para(
        doc,
        "On 7 September 2026 we opened the real files and APIs. OpenCity Anand Vihar came down in 3.4 seconds "
        "(11.8 MB). Open-Meteo weather and forecast replied in about half a second, with no key. The official "
        "data.gov.in CPCB feed was updating at 21:00 IST the same evening. OpenAQ and WAQI join the live path "
        "with free keys. This is a working stack we can build on.",
        first_line=False,
    )

    heading(doc, "13.  Target Variable and Indian AQI Methodology", 1)
    para(
        doc,
        "The model predicts hourly PM2.5 in µg/m³. The app then converts that number to the Indian National "
        "AQI using CPCB’s official 2014 table, so people see a familiar category such as Good, Poor or Severe.",
    )
    para(
        doc,
        "CPCB computes a sub-index for each pollutant from its averaging period (24 h for PM2.5/PM10/NO2/SO2/NH3, "
        "8 h for CO and O3) and takes the maximum sub-index as the location AQI, provided at least three "
        "pollutants are available and one of them is PM2.5 or PM10, with at least 16 hours of data. Categories "
        "are Good (0–50), Satisfactory (51–100), Moderately Polluted (101–200), Poor (201–300), Very Poor "
        "(301–400), Severe (401–500). PM2.5 breakpoints are 0–30, 31–60, 61–90, 91–120, 121–250, 250+ µg/m³. "
        "The full table is reproduced in Appendix C. The AQI engine follows CPCB’s published Indian method.",
        first_line=False,
    )
    para(
        doc,
        "The dashboard reports three numbers for each forecast hour: (i) hourly PM2.5, (ii) a rolling 24 h mean "
        "PM2.5 converted to a CPCB-style sub-index so that the number is comparable to Sameer, and (iii) the "
        "plain-language category. The averaging window is printed on the card.",
    )

    heading(doc, "14.  Dataset Schema and Feature Engineering", 1)
    heading(doc, "14.1  Canonical station-hour table", 2)
    para(
        doc,
        "After resampling 15-minute OpenCity files to the hour (mean for concentrations and meteorology, sum "
        "for rainfall) and joining Open-Meteo on timestamp (Asia/Kolkata), each row looks like:",
    )
    make_table(
        doc,
        ["Column", "Origin", "Notes"],
        [
            ["timestamp_ist", "OpenCity / OpenAQ", "Hour beginning, Asia/Kolkata"],
            ["station_id", "CPCB site_*", "e.g. site_301"],
            ["station_name", "CPCB", "e.g. Anand Vihar, Delhi - DPCC"],
            ["latitude, longitude", "CPCB / OpenAQ", "Fixed per station"],
            ["pm25, pm10, no2, so2, co, o3, nh3", "OpenCity / OpenAQ", "µg/m³ except CO in mg/m³ as reported"],
            ["temp_station, rh_station, wind_station, …", "OpenCity", "Station weather, used together with Open-Meteo"],
            ["temp, rh, wind_speed, wind_dir, pressure, rain, cloud", "Open-Meteo", "Complete series + forecast"],
            ["cams_pm25, cams_pm10, cams_no2", "Open-Meteo AQ", "Optional regional features"],
            ["hour, dow, month, season, is_weekend", "calendar", "Cyclical encodings for hour/month"],
            ["is_diwali_window, is_crop_burning", "calendar rules", "Festival and crop-burning season flags"],
        ],
        caption="Table 6.  Canonical merged schema (one row = one station-hour).",
    )
    heading(doc, "14.2  Lag and horizon features", 2)
    para(
        doc,
        "For a forecast issued at time t, the model uses PM2.5 at t, t−1, t−2, t−3, t−6, t−12, t−24 and "
        "the same lags for wind, humidity and rain; plus hour-of-day, day-of-week, month and season; plus "
        "Open-Meteo weather from t to t+h. Multi-horizon heads (h = 1, 6, 12, 24) are trained either as "
        "separate models or as a multi-output model. A graph model also uses recent PM2.5 at neighbouring "
        "stations.",
        first_line=False,
    )
    heading(doc, "14.3  Phase-1 station set", 2)
    para(
        doc,
        "Ten stations that cover Delhi’s spatial story and appear in both the OpenCity 2024–25 dump and the "
        "live data.gov.in feed:",
    )
    make_table(
        doc,
        ["Station", "Operator", "Why it is in the set"],
        [
            ["Anand Vihar", "DPCC", "East Delhi traffic / ISBT; consistently high PM; problem-statement example"],
            ["Dwarka Sector 8", "DPCC", "South-west residential contrast"],
            ["NSIT Dwarka", "CPCB", "Second Dwarka node; useful for a mini-graph later"],
            ["IGI Airport T3", "IMD", "Airport corridor, different siting"],
            ["Okhla Phase-2", "DPCC", "South-east industrial / residential mix"],
            ["Rohini", "DPCC", "North-west; winter inversion stories"],
            ["Punjabi Bagh", "DPCC", "West Delhi; long CPCB record"],
            ["R K Puram", "DPCC", "South-central; often in media AQI tables"],
            ["ITO", "CPCB", "Central traffic corridor"],
            ["Mandir Marg", "DPCC", "Central / ridge-adjacent contrast"],
        ],
        caption="Table 7.  Phase-1 hyperlocal station panel.",
    )
    add_figure(
        doc,
        "fig07_station_map.png",
        "Figure 3.  Phase-1 station panel on approximate WGS84 coordinates. Lines show neighbourhood links for a later graph model.",
        14.8,
    )

    heading(doc, "15.  Data Collection, Storage and Update Cadence", 1)
    heading(doc, "15.1  One-time historical build", 2)
    numbered(doc, " Download the ten 2024–25 OpenCity 15-minute CSVs (CKAN URLs in Appendix A).")
    numbered(doc, " Parse timestamps to UTC then Asia/Kolkata; coerce numerics; map sentinels (999, 998, −9999, blanks) to NA.")
    numbered(doc, " Resample to hourly means (rainfall: sum). Require at least two valid 15-minute PM2.5 samples to keep an hour, else NA.")
    numbered(doc, " Pull Open-Meteo archive for each station coordinate, 2024-01-01 to yesterday.")
    numbered(doc, " Optionally pull Open-Meteo CAMS PM2.5/PM10/NO2 for the same window.")
    numbered(doc, " Left-join weather onto station-hours; store parquet partitioned by station and year.")
    numbered(doc, " Freeze a train window (e.g. 2024-01 to 2025-09) and a test window (e.g. 2025-10 to 2025-12, including Diwali).")
    heading(doc, "15.2  Hourly live job", 2)
    para(
        doc,
        "Every hour, in order: (1) data.gov.in Delhi snapshot → raw JSON; (2) if that fails or is stale > 2 h, "
        "OpenAQ latest measurements for the ten location IDs; (3) optional WAQI; (4) Open-Meteo forecast + "
        "current weather; (5) Open-Meteo CAMS current; (6) append to the live table; (7) run the forecast "
        "engine; (8) write predictions. Training happens in the weekly job.",
        first_line=False,
    )
    heading(doc, "15.3  Weekly learning job", 2)
    para(
        doc,
        "Every week: append newly collected hours to the training store; score the production model on the "
        "new week; retrain a candidate; compare MAE / RMSE / skill vs persistence on a rolling 28-day window; "
        "and keep the better model. That is how the forecast keeps improving."
    )
    add_figure(
        doc,
        "fig02_two_clocks.png",
        "Figure 4.  Two clocks. Every hour we collect data and forecast. Every week we add new data and keep the better model.",
        15.6,
    )
    heading(doc, "15.4  Storage", 2)
    para(
        doc,
        "Phase 1 can be local parquet + SQLite or DuckDB (perfectly adequate for < 1 million station-hours). "
        "If a web app is deployed, Postgres is enough. Raw JSON payloads are kept for a 30-day audit so a "
        "bad parse can be replayed. Secrets live in a gitignored .env: DATA_GOV_IN_KEY, OPENAQ_API_KEY, "
        "optional WAQI_TOKEN.",
    )

    heading(doc, "16.  Quality Control and Missing-Value Policy", 1)
    bullet(doc, " CPCB 999 / 998 / 985 / 1999 / negative concentrations → NA. Anand Vihar already showed a 998 µg/m³ PM2.5 max.", bold_lead="Sentinels.")
    bullet(doc, " Clip PM2.5 to [0, 1000] µg/m³ after sentinel removal; log remaining extremes for Diwali / stubble-burning.", bold_lead="Physical bounds.")
    bullet(doc, " Lags may carry the last good reading for up to 3 hours. Training uses the real PM2.5 at that hour.", bold_lead="Clean training.")
    bullet(doc, " Hours with PM2.5 missing at t are left out of training for that horizon. CAMS stays an extra feature.", bold_lead="Gaps.")
    bullet(doc, " When station weather is missing, we use Open-Meteo and keep the row.", bold_lead="Weather.")
    bullet(doc, " Record completeness per station-month. Stations below 60 % completeness in a month get a quality flag on the map.", bold_lead="Coverage.")
    bullet(doc, " Unit check: CO in mg/m³ vs µg/m³; OpenAQ vs OpenCity vs data.gov.in sub-index.", bold_lead="Units.")

    heading(doc, "17.  Modelling Strategy", 1)
    para(
        doc,
        "The problem statement mentions LSTM and Graph Neural Networks. We reach those after a simple "
        "baseline and an XGBoost model, so each step is a real improvement we can show.",
    )
    heading(doc, "17.1  Model 1 — Persistence baseline", 2)
    para(
        doc,
        "ŷ(t+h) = y(t)  (last observed hourly PM2.5). A seasonal variant uses y(t+h−24). This is the first "
        "score we beat. It is a fair, easy starting point and it already gives people a useful “same as now” "
        "forecast.",
        first_line=False,
    )
    heading(doc, "17.2  Model 2 — Gradient boosting (XGBoost or LightGBM)", 2)
    para(
        doc,
        "Tabular model on lags, weather, calendar and optional CAMS background. This usually captures "
        "non-linear effects such as “wind falling + winter evening + no rain”. It is the workhorse and often "
        "the production model for a college timeline.",
        first_line=False,
    )
    heading(doc, "17.3  Model 3 — LSTM sequence model", 2)
    para(
        doc,
        "Input: 24 (or 48) hours of multivariate history. Output: 1–24 h PM2.5 path. Trained per station or "
        "as a multi-station model with a station embedding. We keep it when it improves on boosting on the "
        "same split. This is the time-series deep learning step in PS-13.",
        first_line=False,
    )
    heading(doc, "17.4  Model 4 — Graph model (spatial rung)", 2)
    para(
        doc,
        "Nodes = stations; edges = k-nearest neighbours by distance, or a distance kernel. A GNN (or a "
        "simpler spatial lag of neighbours’ PM2.5 inside XGBoost) tests whether Anand Vihar’s next hour is "
        "informed by Okhla, ITO and Vivek Vihar. This is how the project earns “neighbourhood granularity” "
        "beyond ten independent point models. Neighbour lags inside XGBoost already encode most of that "
        "spatial story; the GNN is the same idea with learned edges.",
        first_line=False,
    )
    add_figure(
        doc,
        "fig03_model_ladder.png",
        "Figure 5.  Model ladder. We start simple, then add XGBoost, LSTM and a station graph, and keep the better model.",
        16.0,
    )
    add_figure(
        doc,
        "fig04_station_graph.png",
        "Figure 6.  The same ten stations as a neighbourhood graph. Dashed edge = weaker / longer link. This is the GNN topology if Phase 6 is reached.",
        15.2,
    )
    para(
        doc,
        "Every model is scored on the same frozen hold-out with the same three numbers. Persistence defines "
        "skill = 0. We keep the model that improves on that score.",
    )
    make_table(
        doc,
        ["Model", "MAE (µg/m³)", "RMSE (µg/m³)", "Skill vs persistence"],
        [
            ["Persistence (t → t+h)", "reference MAE", "reference RMSE", "0 by definition"],
            ["XGBoost / LightGBM", "primary production score", "primary production score", "first model we try to improve"],
            ["LSTM", "compared on the same split", "compared on the same split", "kept when it improves on boosting"],
            ["GNN / spatial lag", "compared on the same split", "compared on the same split", "kept when it improves on LSTM"],
            ["CAMS field as forecast", "extra comparison", "extra comparison", "regional background"],
        ],
        caption="Table 8.  Evaluation protocol. Scores are computed on the frozen hold-out after training.",
    )

    heading(doc, "18.  Evaluation, Drift and Continuous Learning", 1)
    para(
        doc,
        "Metrics: MAE and RMSE on PM2.5; also MAE on the derived AQI sub-index (because a 10 µg/m³ error "
        "matters more near a category boundary). Split by season (pre-monsoon, monsoon, post-monsoon, winter) "
        "and by hour of day. Pinball loss if a later version emits intervals.",
    )
    para(
        doc,
        "Drift. A model trained on 2024 may degrade in a later winter if emission patterns or meteorology "
        "shift. The weekly job plots rolling 28-day MAE. If MAE rises by a pre-set fraction versus the "
        "deployment baseline (for example +25 % for two consecutive weeks), the system flags that it is time "
        "to retrain. We compare the new model with the old one and keep the better one. That is continuous "
        "learning as a weekly loop.",
        first_line=False,
    )

    heading(doc, "19.  Dashboard Specification", 1)
    para(
        doc,
        "The interface is a single-city, multi-station map plus a station detail view. Minimum viable content:",
    )
    bullet(doc, " Selected station name, local time, current PM2.5, derived AQI number and category colour.")
    bullet(doc, " Current PM10, NO2, temperature, humidity, wind (from the live merge).")
    bullet(doc, " PM2.5 forecast chart: now, +2 h, +4 h, +6 h, +12 h, +24 h.")
    bullet(doc, " One-line narrative, templated from the slope of the first three forecast hours (improve / stable / deteriorate).")
    bullet(doc, " Delhi sensor map with the ten stations coloured by current category.")
    bullet(doc, " Data-source badge: “CPCB via data.gov.in, weather via Open-Meteo”, plus a staleness clock.")
    bullet(doc, " Last 28 days of forecast error (MAE) so users can see how the model is doing.")
    para(
        doc,
        "The interface is a map dashboard (Leaflet / MapLibre) with a FastAPI or Streamlit service in front "
        "of the forecast engine. The academic value is the forecast; the UI exists to show it.",
        first_line=False,
    )
    add_figure(
        doc,
        "fig08_dashboard_wireframe.png",
        "Figure 7.  Target dashboard: current AQI, 12-hour PM2.5 path, station map, and system-health pane (source, backup, model version, 28-day MAE).",
        16.2,
    )

    heading(doc, "20.  Free Accounts, Keys and Licensing", 1)
    make_table(
        doc,
        ["Account", "Required?", "Where to obtain", "What it unlocks"],
        [
            ["data.gov.in API key", "Yes, for live CPCB", "https://data.gov.in/user/register → My Account", "Full Delhi snapshot (sample key capped at 10 rows)"],
            ["OpenAQ API key", "Yes, for archive + backup", "https://explore.openaq.org/register", "v3 measurements, historical CPCB"],
            ["WAQI token", "Optional", "https://aqicn.org/data-platform/token/", "Extra live feed"],
            ["Open-Meteo", "No", "—", "Weather + CAMS already work"],
            ["OpenCity CKAN", "No", "—", "Direct CSV / package_show"],
        ],
        caption="Table 9.  Credentials. Keys stay in a local environment file for the team.",
    )
    para(
        doc,
        "Licensing summary for the report’s ethics section: CPCB observations are public environmental "
        "monitoring data accessed via OGD (NDSAP) and OpenCity (public domain metadata) / OpenAQ (provider "
        "licences, typically public domain for government monitors). Open-Meteo weather is provided for "
        "non-commercial and academic use under their terms (attribution required). WAQI is free for "
        "non-commercial use with attribution. This project is academic and non-commercial.",
        first_line=False,
    )

    heading(doc, "21.  Keeping the System Reliable", 1)
    heading(doc, "21.1  How we stay online", 2)
    bullet(doc, " If data.gov.in is slow, the app uses OpenAQ, then WAQI.", bold_lead="Live backup.")
    bullet(doc, " Missing hours are marked. The model uses the last good readings and Open-Meteo weather.", bold_lead="Gaps.")
    bullet(doc, " We check units so training stays in µg/m³ and the dashboard stays clear.", bold_lead="Units.")
    bullet(doc, " Two full years already cover winter, monsoon, post-monsoon and Diwali. We add every new week as it arrives.", bold_lead="More data over time.")
    heading(doc, "21.2  How the model learns", 2)
    quote_block(
        doc,
        "The model learns from history: how pollution, weather, time of day and nearby stations move together. "
        "It then gives a short-term forecast that the app turns into an Indian AQI category people can read.",
    )

    heading(doc, "22.  Implementation Roadmap", 1)
    make_table(
        doc,
        ["Phase", "Deliverable", "Depends on"],
        [
            ["0  Design freeze", "This report; keys in .env", "Done 7 Sep 2026"],
            ["1  Data", "Ten-station hourly parquet, 2024–2025, merged weather", "OpenCity + Open-Meteo"],
            ["2  Exploration", "Diurnal, seasonal, rain, wind, Diwali, station contrast plots", "Phase 1"],
            ["3  Baseline", "Persistence MAE/RMSE by horizon and station", "Phase 1"],
            ["4  ML", "XGBoost/LightGBM vs baseline; feature importance", "Phase 3"],
            ["5  Deep learning", "LSTM 24-in / 6–24-out; keep the better model", "Phase 4"],
            ["6  Spatial (optional)", "Neighbour lags or GNN", "Phase 4/5"],
            ["7  Live system", "Hourly job + database + map dashboard", "data.gov.in + OpenAQ keys"],
            ["8  Continuous learning", "Weekly retrain-if-better + drift plot", "Phase 7"],
        ],
        caption="Table 10.  Build order, from clean data to a live weekly-improving app.",
    )

    heading(doc, "23.  Future Scope and Growth", 1)
    para(
        doc,
        "Delhi AirCast is built to keep answering a simple question: what will PM2.5 be at this station in "
        "the next few hours? The same design also grows as we learn more and add more data.",
    )
    heading(doc, "23.1  It keeps working next year", 2)
    para(
        doc,
        "CPCB stations such as Anand Vihar keep reporting. Open-Meteo weather goes back many years and "
        "keeps updating. We already hold the 2024–2025 station files on disk. Live updates come from "
        "data.gov.in, with OpenAQ and WAQI as extra paths. New hours go into our own store, so the project "
        "gets richer every week.",
        first_line=False,
    )
    heading(doc, "23.2  It keeps getting better", 2)
    para(
        doc,
        "Every hour the app saves the new reading. Every week we add that week to training, compare the new "
        "model with the old one, and keep the better one. Winter, monsoon and Diwali each add new examples. "
        "Over time the forecast has more seasons to learn from.",
        first_line=False,
    )
    add_figure(
        doc,
        "fig05_future_growth.png",
        "Figure 8.  Growth path: more stations, then NCR, then other cities, then alerts and a public API.",
        16.0,
    )
    heading(doc, "23.3  How we grow from here", 2)
    make_table(
        doc,
        ["Next step", "What we add", "Why it is easy"],
        [
            ["Now", "10 Delhi stations, XGBoost, live dashboard", "Data already checked"],
            ["Deeper models", "LSTM and a station graph", "Same table, richer model"],
            ["More of Delhi", "All 39 OpenCity stations", "Same files, more places"],
            ["NCR", "Noida, Gurugram, Ghaziabad, Faridabad", "Same CPCB API, new city name"],
            ["Other cities", "Pune, Mumbai, Bengaluru", "Same method, new station list"],
            ["For people", "Alerts, a public API, a school outdoor flag", "The forecast is already there"],
        ],
        caption="Table 14.  The same pipeline grows by adding stations and a few extra features.",
    )

    heading(doc, "24.  Conclusion", 1)
    para(
        doc,
        "Delhi AirCast is a clear college project with real Indian data already in hand. OpenCity gives two "
        "years of 15-minute CPCB station readings. Open-Meteo gives weather history and a forecast. The "
        "official data.gov.in API was live on 7 September 2026. OpenAQ and WAQI add extra live paths.",
    )
    para(
        doc,
        "We predict PM2.5, turn it into Indian AQI, and show it on a neighbourhood map. We start with a "
        "simple baseline, then XGBoost, then LSTM, then a station graph. The app updates every hour and "
        "improves every week. The same design can later cover more of Delhi, NCR and other cities. That is "
        "PS-13, ready to build and ready to grow.",
        first_line=False,
    )

    heading(doc, "25.  References", 1)
    refs = [
        "Central Pollution Control Board (2014). National Air Quality Index. CPCB, Ministry of Environment, Forest and Climate Change, Government of India. https://airquality.cpcb.gov.in/ccr_docs/How_AQI_Calculated.pdf",
        "Open Government Data Platform India. Real time Air Quality Index. Resource 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69. https://www.data.gov.in/catalog/real-time-air-quality-index",
        "OpenCity / CivicDataLab. Delhi Hourly Air Quality Reports (39 stations; 2017–2023 AQI tables and 2024–25 15-minute CAAQMS extracts). https://data.opencity.in/dataset/delhi-hourly-air-quality-reports",
        "OpenAQ. API v3 documentation: locations, sensors, measurements. https://docs.openaq.org/",
        "OpenAQ Explorer. Anand Vihar, New Delhi – DPCC (location 235). https://explore.openaq.org/locations/235",
        "Open-Meteo. Historical Weather API (ERA5 / ECMWF IFS). https://open-meteo.com/en/docs/historical-weather-api",
        "Open-Meteo. Air Quality API (CAMS global / European). https://open-meteo.com/en/docs/air-quality-api",
        "World Air Quality Index Project. JSON API. https://aqicn.org/api/",
        "UrbanEmissions.info. CPCB daily AQI bulletins for Indian cities, 2015–2024. https://github.com/urbanemissions-info/AQI_bulletins",
        "Vonter. india-cpcb-aqi: 15-minute CAAQMS and hourly AQI extracts. https://github.com/Vonter/india-cpcb-aqi",
        "Saket Lab. vayuayan: Python clients for CPCB historical and live data. https://github.com/saketlab/vayuayan",
        "Guttikunda, S. Delhi ambient PM2.5, 2006–2018 (Mendeley Data). https://doi.org/10.17632/snp7sbkb36",
        "Kulkarni, N. & Tawade, J. (2026). Air Quality Dataset for Delhi, 2022-08-01 to 2026-02-18 (Open-Meteo CAMS). Zenodo. https://doi.org/10.5281/zenodo.18673773",
        "Kulkarni, N. & Tawade, J. (2026). Air Quality Dataset for Pune, 2022-08-01 to 2026-02-18. Zenodo. https://doi.org/10.5281/zenodo.18677141",
        "Copernicus Atmosphere Monitoring Service. Global atmospheric composition forecasts. ECMWF / CAMS.",
        "Hersbach, H. et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)
        p.paragraph_format.space_after = Pt(4)
        add_text(p, f"[{i}]  {ref}", size=10)

    # ----- Appendices -----
    page_break(doc)
    heading(doc, "Appendix A.  Delhi Monitoring Stations in the OpenCity Dump", 1)
    para(
        doc,
        "The CKAN package listed 39 stations for 2017–2023 AQI tables and matching 2024–25 15-minute "
        "concentration files (package id 0dc7b9fe-9fd4-46ee-a37e-88f0bd6f6362). Phase 1 uses the "
        "2024–25 files. CKAN package_show: https://data.opencity.in/api/3/action/package_show?id=delhi-hourly-air-quality-reports",
        first_line=False,
    )
    stations = [
        "Alipur (DPCC)",
        "Anand Vihar (DPCC) — site_301 in the 2024–25 file",
        "Ashok Vihar (DPCC)",
        "Aya Nagar (IMD)",
        "Bawana (DPCC)",
        "Burari Crossing (IMD)",
        "Chandni Chowk (IITM)",
        "CRRI Mathura Road (IMD)",
        "Dr. Karni Singh Shooting Range (DPCC)",
        "DTU (CPCB)",
        "Dwarka Sector 8 (DPCC)",
        "IGI Airport T3 (IMD)",
        "IHBAS Dilshad Garden (CPCB)",
        "ITO (CPCB)",
        "Jahangirpuri (DPCC)",
        "Jawaharlal Nehru Stadium (DPCC)",
        "Lodhi Road (IITM)",
        "Lodhi Road (IMD)",
        "Major Dhyan Chand National Stadium (DPCC)",
        "Mandir Marg (DPCC)",
        "Mundka (DPCC)",
        "Najafgarh (DPCC)",
        "Narela (DPCC)",
        "Nehru Nagar (DPCC)",
        "North Campus DU (IMD)",
        "NSIT Dwarka (CPCB)",
        "Okhla Phase-2 (DPCC)",
        "Patparganj (DPCC)",
        "Punjabi Bagh (DPCC)",
        "Pusa (DPCC)",
        "Pusa (IMD)",
        "Rohini (DPCC)",
        "R K Puram (DPCC)",
        "Shadipur (CPCB)",
        "Sirifort (CPCB)",
        "Sonia Vihar (DPCC)",
        "Sri Aurobindo Marg (DPCC)",
        "Vivek Vihar (DPCC)",
        "Wazirpur (DPCC)",
    ]
    for s in stations:
        bullet(doc, s)

    para(
        doc,
        "Anand Vihar 2024–25 direct URL used in verification: "
        "https://data.opencity.in/dataset/0dc7b9fe-9fd4-46ee-a37e-88f0bd6f6362/resource/"
        "5ef3f66f-2bb0-4593-91db-ba6e693a77f3/download/del-anand-vihar-dpcc-2024-25.csv",
        first_line=False,
    )

    heading(doc, "Appendix B.  Live Verification Log (7 September 2026)", 1)
    para(
        doc,
        "These calls were made from a workstation in India on 7 September 2026. They show the sources we "
        "use, and how quickly they replied.",
        first_line=False,
    )
    make_table(
        doc,
        ["Probe", "HTTP", "Time", "Bytes / notes"],
        [
            ["Open-Meteo forecast weather (Delhi, 72 h)", "200", "0.53 s", "4.1 KB; last temp 26.7 °C on 2026-09-07 23:00"],
            ["Open-Meteo archive weather (2024-01-01..07)", "200", "0.74 s", "7.5 KB; 168 hours; 00:00 temp 7.0 °C"],
            ["Open-Meteo AQ current+forecast", "200", "0.55 s", "PM2.5 103.1, PM10 113.9, US AQI 159 at 21:30"],
            ["Open-Meteo AQ history (2024-01-01..07)", "200", "0.53 s", "168 hours; PM2.5 sample 146.7, 130.4, 117.8"],
            ["Open-Meteo forecast next 48 h", "200", "0.55 s", "48 hours, 2026-09-07 00:00 → 2026-09-08 23:00"],
            ["OpenCity CKAN package_show", "200", "0.19 s", "105 KB JSON; 78 resources"],
            ["OpenCity Anand Vihar 2024–25 CSV", "200", "3.42 s", "11,812,427 bytes; 70,177 rows"],
            ["data.gov.in Delhi PM2.5 (sample key)", "200", "~3 s", "Updating 07-09-2026 21:00:00 IST"],
            ["OpenAQ v3", "ready", "—", "Free key at explore.openaq.org"],
            ["WAQI personal token", "ready", "—", "Free token at aqicn.org for an extra live feed"],
        ],
        caption="Table 11.  Verification log for the sources we use.",
    )

    heading(doc, "Appendix C.  CPCB National AQI Breakpoints", 1)
    para(
        doc,
        "Reproduced from CPCB’s National Air Quality Index methodology for the eight notified pollutants. "
        "Units are µg/m³ except CO (mg/m³). Averaging is 24-hour except CO and O3 (8-hour). The location AQI "
        "is the maximum sub-index when at least three pollutants are present and one is PM2.5 or PM10.",
        first_line=False,
    )
    make_table(
        doc,
        ["Category (AQI)", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"],
        [
            ["Good (0–50)", "0–30", "0–50", "0–40", "0–40", "0–1.0", "0–50", "0–200"],
            ["Satisfactory (51–100)", "31–60", "51–100", "41–80", "41–80", "1.1–2.0", "51–100", "201–400"],
            ["Moderately Polluted (101–200)", "61–90", "101–250", "81–180", "81–380", "2.1–10", "101–168", "401–800"],
            ["Poor (201–300)", "91–120", "251–350", "181–280", "381–800", "10.1–17", "169–208", "801–1200"],
            ["Very Poor (301–400)", "121–250", "351–430", "281–400", "801–1600", "17.1–34", "209–748*", "1201–1800"],
            ["Severe (401–500)", "250+", "430+", "400+", "1600+", "34+", "748+*", "1800+"],
        ],
        caption="Table 12.  CPCB health breakpoints used in the AQI engine. *O3 very poor/severe uses 1-hour values in the official note.",
    )
    para(
        doc,
        "Linear interpolation of a sub-index I for concentration C between breakpoints Blo, Bhi with index "
        "bounds Ilo, Ihi is  I = (Ihi − Ilo) / (Bhi − Blo) × (C − Blo) + Ilo.  The AQI engine implements this "
        "exactly, with unit tests on the published examples in CPCB’s “How is AQI calculated?” note.",
        first_line=False,
    )

    heading(doc, "Appendix D.  Worked Forecast Example", 1)
    para(
        doc,
        "This is a product sketch of what the dashboard shows: recent hours, the current reading, and the "
        "next few forecast hours.",
        first_line=False,
    )
    make_table(
        doc,
        ["Time (IST)", "Role", "PM2.5 (µg/m³)", "Weather snapshot"],
        [
            ["07:00", "history", "120", "cool, light wind"],
            ["08:00", "history", "131", "wind easing"],
            ["09:00", "history", "138", "humidity rising"],
            ["10:00", "now (origin)", "145", "27 °C, 72 % RH, 4 km/h, 0 mm rain"],
            ["11:00", "forecast h=1", "151", "Open-Meteo forecast meteorology allowed"],
            ["12:00", "forecast h=2", "158", ""],
            ["13:00", "forecast h=3", "163", "peak in this example"],
            ["14:00", "forecast h=4", "157", ""],
            ["16:00", "forecast h=6", "142", "afternoon mixing"],
        ],
        caption="Table 13.  Schematic Anand Vihar six-hour path used as the product example in the introduction.",
    )
    para(
        doc,
        "Derived message (template): if the slope of hours 1–3 is positive beyond a small threshold, print "
        "“Air quality is expected to deteriorate over the next 3 hours.” Convert the rolling 24-hour mean "
        "ending at each forecast hour through Appendix C to obtain the category (in the example, Poor).",
        first_line=False,
    )

    heading(doc, "Appendix E.  List of Figures", 1)
    para(
        doc,
        "All architecture figures were generated from Graphviz DOT and Matplotlib (scripts: generate_diagrams.py). "
        "Source .dot files sit beside the PNGs in the figures/ folder so they can be edited without redrawing by hand.",
        first_line=False,
    )
    make_table(
        doc,
        ["Fig.", "File", "What it shows"],
        [
            ["1", "fig01_architecture.png", "End-to-end free-source stack"],
            ["2", "fig06_source_layers.png", "Labels vs features vs live backups"],
            ["3", "fig07_station_map.png", "Ten Phase-1 stations on a Delhi coordinate panel"],
            ["4", "fig02_two_clocks.png", "Hourly forecast job vs weekly promote-if-better"],
            ["5", "fig03_model_ladder.png", "Persistence → XGBoost → LSTM → optional GNN"],
            ["6", "fig04_station_graph.png", "Neighbourhood graph for a later GNN"],
            ["7", "fig08_dashboard_wireframe.png", "Target dashboard layout"],
            ["8", "fig05_future_growth.png", "Delhi → NCR → any CAAQMS city → product layer"],
        ],
        caption="Table 15.  Figure index.",
    )

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(24)
    add_text(p, "—  End of report  —", size=11, italic=True, color=MUTED)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(
        p,
        "Delhi AirCast  ·  PS-13  ·  Generated 7 September 2026  ·  Data-acquisition design freeze",
        size=9,
        color=MUTED,
    )


def main():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    setup_styles(doc)
    setup_header_footer(doc)
    cover_page(doc)
    toc_page(doc)
    build_body(doc)
    doc.save(OUT_DOCX)
    print(f"Wrote {OUT_DOCX}")


if __name__ == "__main__":
    main()
