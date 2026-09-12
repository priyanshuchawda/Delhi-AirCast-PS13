#!/usr/bin/env python3
"""Short, clear review PPT for PS-13 Delhi AirCast."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "Delhi_AirCast_PS13_Presentation.pptx"

NAVY = RGBColor(0x1B, 0x3A, 0x5F)
ACCENT = RGBColor(0x2C, 0x5F, 0x8A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x5B, 0x6B, 0x7A)
PALE = RGBColor(0xEE, 0xF3, 0xF8)

W = Inches(13.333)
H = Inches(7.5)
TOTAL = 10


def set_run(run, *, size=18, bold=False, italic=False, color=DARK, font="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font


def add_text_box(slide, l, t, w, h, text, *, size=18, bold=False, italic=False, color=DARK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_run(run, size=size, bold=bold, italic=italic, color=color)
    return box


def add_bullets(slide, l, t, w, h, items, *, size=20):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        run = p.add_run()
        run.text = "•  " + item
        set_run(run, size=size, color=DARK)
    return box


def footer(slide, n):
    add_text_box(slide, Inches(0.4), Inches(7.15), Inches(10), Inches(0.28), "PS-13  ·  Delhi AirCast", size=11, color=MUTED)
    add_text_box(slide, Inches(11.4), Inches(7.15), Inches(1.5), Inches(0.28), f"{n}  /  {TOTAL}", size=11, color=MUTED, align=PP_ALIGN.RIGHT)


def title_bar(slide, title):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.95))
    sh.fill.solid()
    sh.fill.fore_color.rgb = NAVY
    sh.line.fill.background()
    add_text_box(slide, Inches(0.45), Inches(0.24), Inches(12.4), Inches(0.5), title, size=26, bold=True, color=WHITE)


def pic(slide, name, l, t, w):
    path = FIG / name
    if path.exists():
        slide.shapes.add_picture(str(path), l, t, width=w)


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def s01(prs):
    s = blank(prs)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.75), W, Inches(1.75))
    accent.fill.solid()
    accent.fill.fore_color.rgb = ACCENT
    accent.line.fill.background()
    add_text_box(s, Inches(0.7), Inches(1.45), Inches(12), Inches(0.4), "PS-13  ·  AI-Powered Air Quality Forecasting", size=16, bold=True, color=RGBColor(0xB8, 0xD0, 0xE8))
    add_text_box(s, Inches(0.7), Inches(2.0), Inches(12), Inches(1.0), "Delhi AirCast", size=48, bold=True, color=WHITE)
    add_text_box(s, Inches(0.7), Inches(3.2), Inches(12), Inches(1.1), "A simple forecast for your neighbourhood:\nPM2.5 and AQI for the next 1 to 24 hours", size=22, color=WHITE)
    add_text_box(
        s,
        Inches(0.7),
        Inches(6.05),
        Inches(12),
        Inches(1.2),
        "Priyanshu Chawda  ·  Aryan Babel  ·  Shruti Agrawal  ·  Aditya Gayal\n"
        "Undergraduate project  ·  September 2026",
        size=18,
        color=WHITE,
    )


def s02(prs):
    s = blank(prs)
    title_bar(s, "Why this project")
    add_bullets(
        s,
        Inches(0.6),
        Inches(1.4),
        Inches(12),
        Inches(5.3),
        [
            "People in Delhi need to know the air in their own neighbourhood.",
            "They also need the next few hours, so they can plan a walk, a run, or school pickup.",
            "We build Delhi AirCast: a short-term air forecast for each monitoring station.",
            "The app shows today’s air, the next 6–24 hours, and a clear line in simple words.",
        ],
    )
    footer(s, 2)


def s03(prs):
    s = blank(prs)
    title_bar(s, "What the user sees")
    cards = [
        (0.45, "Right now", "Anand Vihar, 10:00\n\nPM2.5   145 µg/m³\nAQI about 187\nCategory: Poor"),
        (4.7, "Next 6 hours", "11 AM   151\n12 PM   158\n1 PM    163\n2 PM    157\n4 PM    142"),
        (8.95, "In simple words", "Air quality may get\nworse over the next\n3 hours.\n\nThen it eases a little\nin the afternoon."),
    ]
    for x, title, body in cards:
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.45), Inches(3.95), Inches(5.1))
        sh.fill.solid()
        sh.fill.fore_color.rgb = PALE
        sh.line.color.rgb = NAVY
        add_text_box(s, Inches(x + 0.25), Inches(1.7), Inches(3.45), Inches(0.5), title, size=20, bold=True, color=NAVY)
        add_text_box(s, Inches(x + 0.25), Inches(2.35), Inches(3.45), Inches(3.9), body, size=20, color=DARK)
    footer(s, 3)


def s04(prs):
    s = blank(prs)
    title_bar(s, "Our data — all free, already working")
    add_bullets(
        s,
        Inches(0.6),
        Inches(1.4),
        Inches(12),
        Inches(5.3),
        [
            "Pollution: CPCB / DPCC station data from OpenCity (15-minute files for 2024–2025).",
            "Live updates: official data.gov.in CPCB API (stations were updating on 7 Sep 2026).",
            "Weather: Open-Meteo — history and the next 48 hours, no API key.",
            "Backup: OpenAQ, the same government stations through a second free API.",
            "We checked these sources ourselves. They respond in a few seconds.",
        ],
    )
    footer(s, 4)


def s05(prs):
    s = blank(prs)
    title_bar(s, "How the system fits together")
    pic(s, "fig01_architecture.png", Inches(0.55), Inches(1.15), Inches(12.2))
    footer(s, 5)


def s06(prs):
    s = blank(prs)
    title_bar(s, "How we forecast")
    add_bullets(
        s,
        Inches(0.6),
        Inches(1.35),
        Inches(12),
        Inches(2.4),
        [
            "The model predicts PM2.5. The app then converts that number to Indian AQI.",
            "We compare three models and keep the one that works best.",
        ],
        size=20,
    )
    pic(s, "fig03_model_ladder.png", Inches(0.5), Inches(3.5), Inches(12.3))
    footer(s, 6)


def s07(prs):
    s = blank(prs)
    title_bar(s, "Ten stations across Delhi")
    pic(s, "fig07_station_map.png", Inches(0.3), Inches(1.15), Inches(7.5))
    add_bullets(
        s,
        Inches(8.05),
        Inches(1.5),
        Inches(4.8),
        Inches(5.2),
        [
            "Anand Vihar, ITO, Okhla",
            "Dwarka, NSIT, IGI Airport",
            "Rohini, Punjabi Bagh",
            "RK Puram, Mandir Marg",
            "Each place gets its own forecast.",
            "Later we can add more stations and nearby NCR cities.",
        ],
        size=18,
    )
    footer(s, 7)


def s08(prs):
    s = blank(prs)
    title_bar(s, "The dashboard")
    pic(s, "fig08_dashboard_wireframe.png", Inches(0.55), Inches(1.12), Inches(12.2))
    footer(s, 8)


def s09(prs):
    s = blank(prs)
    title_bar(s, "Our plan, and how we grow")
    steps = [
        ("1", "Clean data", "Merge pollution\nand weather"),
        ("2", "First models", "Simple baseline,\nthen XGBoost"),
        ("3", "Deeper AI", "LSTM, then a\nstation graph"),
        ("4", "Live app", "Hourly update\nand weekly improve"),
    ]
    for i, (n, title, note) in enumerate(steps):
        x = 0.45 + i * 3.2
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.4), Inches(3.0), Inches(2.55))
        sh.fill.solid()
        sh.fill.fore_color.rgb = PALE
        sh.line.color.rgb = NAVY
        add_text_box(s, Inches(x + 0.18), Inches(1.55), Inches(2.65), Inches(0.4), n, size=16, bold=True, color=ACCENT)
        add_text_box(s, Inches(x + 0.18), Inches(2.0), Inches(2.65), Inches(0.55), title, size=20, bold=True, color=NAVY)
        add_text_box(s, Inches(x + 0.18), Inches(2.65), Inches(2.65), Inches(1.0), note, size=16, color=DARK)
    add_bullets(
        s,
        Inches(0.6),
        Inches(4.2),
        Inches(12),
        Inches(2.6),
        [
            "Every hour the app collects new readings and shows a fresh forecast.",
            "Every week we add the new data and keep the better model.",
            "The same design can later cover more Delhi stations, NCR, and other cities such as Pune.",
        ],
        size=18,
    )
    footer(s, 9)


def s10(prs):
    s = blank(prs)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    add_text_box(s, Inches(0.7), Inches(1.9), Inches(12), Inches(0.9), "Thank you", size=44, bold=True, color=WHITE)
    add_text_box(s, Inches(0.7), Inches(3.0), Inches(12), Inches(1.0), "Delhi AirCast  ·  a clear air forecast for every neighbourhood", size=20, color=WHITE)
    add_text_box(
        s,
        Inches(0.7),
        Inches(4.4),
        Inches(12),
        Inches(1.4),
        "Priyanshu Chawda  ·  Aryan Babel\nShruti Agrawal  ·  Aditya Gayal",
        size=22,
        bold=True,
        color=RGBColor(0xB8, 0xD0, 0xE8),
    )


def main():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    s01(prs)
    s02(prs)
    s03(prs)
    s04(prs)
    s05(prs)
    s06(prs)
    s07(prs)
    s08(prs)
    s09(prs)
    s10(prs)
    prs.save(OUT)
    print("Wrote", OUT, "slides", len(prs.slides))


if __name__ == "__main__":
    main()
