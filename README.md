# Delhi AirCast — PS-13

College report + review slides for **PS-13 AI-Powered Air Quality Forecasting**.

## Deliverables

| File | What |
|------|------|
| `Delhi_AirCast_PS13_Data_Acquisition_Report.docx` | Full report (edit cover: name, roll, college) |
| `Delhi_AirCast_PS13_Data_Acquisition_Report.pdf` | Printable PDF |
| `Delhi_AirCast_PS13_Presentation.pptx` | Review deck |
| `Delhi_AirCast_PS13_Presentation.pdf` | Slides as PDF |
| `figures/` | Architecture diagrams (Graphviz + Matplotlib) |

## Regenerate

```bash
cd ~/Delhi-AirCast-PS13
uv run python generate_diagrams.py
uv run python generate_report.py
uv run python generate_ppt.py
soffice --headless --convert-to pdf:writer_pdf_Export Delhi_AirCast_PS13_Data_Acquisition_Report.docx
soffice --headless --convert-to pdf Delhi_AirCast_PS13_Presentation.pptx
```

Fill the cover-page blanks before submitting.

## Acquire public datasets

Downloaded data is intentionally excluded from Git. Keep the raw files under
`data/raw/` and retain the generated manifests and source metadata.

```bash
uv run python scripts/download_datasets.py --source all
```

The downloader currently collects the 78-resource OpenCity/CPCB Delhi archive,
central-Delhi historical weather, CAMS air-quality history, and CPCB reference
documents. Large research mirrors are downloaded separately when their public
hosting requires a dedicated client.

## Validate the AQI engine

```bash
uv run pytest -q
```

The CPCB implementation returns a clearly labelled PM2.5-only proxy when a
station does not have enough pollutant inputs for an official overall AQI. It
does not silently turn a single PM2.5 measurement into a full AQI.
