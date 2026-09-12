# Delhi AirCast data sources

This project keeps each source separate until station identity, timestamps,
units, coverage, and provenance have been checked. A downloaded file is not
automatically treated as ground truth.

| Source | Role | Access | Local status |
|---|---|---|---|
| [OpenCity Delhi Hourly Air Quality Reports](https://data.opencity.in/dataset/delhi-hourly-air-quality-reports) | Primary CPCB/DPCC historical station archive; 39 stations and 78 CSV resources | Public | Downloaded under `data/raw/opencity/` |
| [CPCB real-time AQI on data.gov.in](https://www.data.gov.in/resource/real-time-air-quality-index-various-locations) | Official live station snapshot | Free API key required for full API use | Resumable connector in `scripts/download_cpcb_live.py`; key not stored in Git |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Historical meteorological features | Public, no key for this academic use | Downloaded under `data/raw/weather/` |
| [Open-Meteo Air Quality / CAMS](https://open-meteo.com/en/docs/air-quality-api) | Regional auxiliary PM and gas features | Public, no key for this academic use | Downloaded under `data/raw/cams/` |
| [IIT Delhi AirDelhi](https://www.cse.iitd.ac.in/pollutiondata/download) | Mobile-sensor, fine-grained spatial benchmark | Public Google Drive; CC BY 4.0 | Partial local capture under `data/raw/airdelhi/`; complete mirror under `data/raw/huggingface/DelhiPollDataset/` |
| [AirDelhi Hugging Face mirror](https://huggingface.co/datasets/sachin-iitd/DelhiPollDataset) | Raw/clean/grid benchmark variants | Public; CC BY 4.0 | Downloaded under `data/raw/huggingface/` |
| [Zenodo India PM2.5 dataset](https://zenodo.org/records/20789755) | Processed 3-hourly multi-city comparison dataset | Public download | Downloaded under `data/raw/zenodo/` |
| [OpenAQ API](https://docs.openaq.org/api) | Historical/live backup and station cross-check | Free API key required | Catalog + latest batches downloaded under `data/raw/openaq/`; hourly batches resumable |
| [OpenAQ AWS archive](https://docs.openaq.org/aws/about) | No-account daily CSV archive for known location IDs | Public S3 | Location coverage being audited |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/active_fire/) | MODIS/VIIRS active-fire hotspots for biomass-burning and regional-event features | Free MAP_KEY required for API | Resumable connector in `scripts/download_firms.py`; raw data is local-only |
| [Kaggle Delhi datasets](https://www.kaggle.com/datasets/jatinkalra17/delhi-aqi) | Community comparison datasets only | Kaggle credentials/token generally required | Ten curated datasets downloaded under `data/raw/kaggle/` |

## Data policy

- Raw files remain source-specific and are not silently merged.
- CPCB station data is the primary supervised target for the product.
- AirDelhi's 1 km x 1 hour grid is a spatial/mobile-sensor benchmark and is
  normalized separately; it is not treated as interchangeable with CPCB
  reference-station labels.
- AirDelhi mobile sensors are useful for spatial research and interpolation,
  but are not interchangeable with CPCB reference-station labels.
- CAMS is an auxiliary regional model field, not a station observation.
- NASA FIRMS is an auxiliary event/source-proxy layer; active-fire detections
  are not direct measurements of Delhi surface PM2.5.
- Community Kaggle/Hugging Face derivatives are comparison inputs only until
  their provenance and transformations are audited.
- API keys belong in a local `.env` file and must never be committed.
