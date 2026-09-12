# Kaggle comparison sources

These public datasets are downloaded for schema comparison, historical
coverage, and robustness checks. CPCB/OpenCity remains the primary reference
station target; community Kaggle derivatives are not silently merged into it.

| Dataset | Intended use | License reported by Kaggle |
|---|---|---|
| `rohanrao/air-quality-data-in-india` | Large 2015–2020 India archive; Delhi slice comparison | CC0-1.0 |
| `abhisheksjha/time-series-air-quality-data-of-india-2010-2023` | Long historical India time series | CC-BY-NC-SA-4.0 |
| `aniket0712/delhi-ncr-hourly-air-quality-dataset-20192024` | Delhi/NCR hourly comparison | CC0-1.0 |
| `kunshbhatia/delhi-air-quality-dataset` | Small Delhi tabular comparison | DbCL-1.0 |
| `deepaksirohiwal/delhi-air-quality` | Small Delhi comparison | CC-BY-NC-SA-4.0 |
| `anuragbantu/new-delhi-air-quality` | Small New Delhi comparison | CC0-1.0 |
| `digantdixit/delhi-air-quality-index-data` | AQI label comparison | CC0-1.0 |
| `jatinkalra17/delhi-aqi` | PM2.5/AQI comparison through 2025 | MIT |
| `sumanbera19/delhi-air-quality-dataset-20232025-aqi-data` | Recent AQI comparison | CC-BY-SA-4.0 |
| `sohails07/delhi-weather-and-aqi-dataset-2025` | Weather/AQI feature comparison | MIT |

Download them locally with:

```bash
uv run python scripts/download_kaggle.py
```

The script uses the Kaggle CLI's external authentication and never stores a
token in the repository. Downloaded files are ignored by Git. Review each
dataset's license and provenance before using it in a published model.
