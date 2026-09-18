# Crop Yield Prediction

An end-to-end crop yield prediction project built from scratch for real agriculture data workflows. The project combines:

- Government crop production data as the primary target source.
- Kaggle rainfall and crop-yield data as supplemental context.
- Live WeatherAPI data for real-time inference.
- SoilGrids soil-property enrichment.
- Sentinel-2 NDVI extraction through Microsoft Planetary Computer.
- Model comparison across Linear Regression, Random Forest, and XGBoost.
- Hyperparameter tuning with `GridSearchCV`.
- Feature engineering for time, rainfall, and historical-yield patterns.

## Project layout

```text
.
├── artifacts/
├── data/
├── scripts/
├── src/crop_yield_prediction/
└── tests/
```

## Data sources

The project is wired for these real sources:

1. Government crop production data:
   `https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0`
2. Kaggle supplemental dataset:
   `chinmaynagesh/crop-yield-per-state-and-rainfall-data-of-india`
3. Weather API:
   `https://www.weatherapi.com/`
4. SoilGrids REST API:
   `https://dev-rest.isric.org/soilgrids/v2.0/docs`
5. Planetary Computer STAC:
   `https://planetarycomputer.microsoft.com/api/stac/v1`

## Setup

Use Python `3.11`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[api,geo,dev]"
```

Create a local `.env` file from `.env.example` and add your own WeatherAPI key. Kaggle downloads require a `kaggle.json` token, and this project looks for it in a local workspace folder:

```text
.kaggle/kaggle.json
```

## Bootstrapping data

```bash
python3.11 scripts/bootstrap_data.py
```

What the bootstrap script does:

- Tries to download the Kaggle dataset through the Kaggle CLI.
- Downloads the government dataset only if you set a direct `GOV_DATASET_URL`.
- Creates clean raw-data folders either way so training can proceed once files are present.

Before running the Kaggle bootstrap, place your Kaggle API token at `.kaggle/kaggle.json`.

If the OGD portal gives you a downloadable CSV or ZIP link, place it inside `data/raw/government/`.

## Training

```bash
python3.11 scripts/train_model.py
```

Outputs:

- `artifacts/models/best_model.joblib`
- `artifacts/metrics/model_comparison.csv`
- `artifacts/metadata/feature_columns.json`
- `artifacts/metadata/reference_history.csv`

## Live prediction API

```bash
python3.11 scripts/run_api.py
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict/live \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "Rice",
    "state": "Karnataka",
    "district": "Mysuru",
    "season": "Kharif",
    "area_ha": 2.0,
    "latitude": 12.2958,
    "longitude": 76.6394
  }'
```

## Notes

- The training code is schema-tolerant and tries to normalize common Indian crop-data column names.
- NDVI support needs the `geo` extra installed.
- If soil or satellite calls fail, the live API still predicts with the remaining features.
- Historical weather backfills are not included yet; current weather is used for live inference only.
