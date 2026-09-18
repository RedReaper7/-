from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import predict_live

app = FastAPI(title="Crop Yield Prediction API", version="0.1.0")

HOME_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crop Yield Prediction</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, sans-serif;
      background: #f5f7f3;
      color: #1d2a22;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 32px 16px;
    }
    main {
      width: min(920px, 100%);
      background: #ffffff;
      border: 1px solid #d9e1d6;
      border-radius: 8px;
      box-shadow: 0 12px 32px rgba(34, 48, 38, 0.12);
      padding: 28px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 28px;
    }
    p {
      margin: 0 0 22px;
      color: #536157;
    }
    form {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 14px;
      font-weight: 700;
    }
    input {
      min-width: 0;
      border: 1px solid #bcc8bb;
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
    }
    button {
      grid-column: 1 / -1;
      border: 0;
      border-radius: 6px;
      background: #236341;
      color: #ffffff;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      padding: 12px 16px;
    }
    pre {
      min-height: 44px;
      margin: 20px 0 0;
      overflow: auto;
      border-radius: 6px;
      background: #132018;
      color: #edf7ef;
      padding: 14px;
      white-space: pre-wrap;
    }
    @media (max-width: 680px) {
      form {
        grid-template-columns: 1fr;
      }
      main {
        padding: 22px;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Crop Yield Prediction</h1>
    <p>Enter field details to call the live prediction API.</p>
    <form id="prediction-form">
      <label>Crop <input name="crop" value="Rice" required></label>
      <label>State <input name="state" value="Karnataka"></label>
      <label>District <input name="district" value="Mysuru"></label>
      <label>Season <input name="season" value="Kharif"></label>
      <label>Area ha <input name="area_ha" type="number" step="0.01" value="2"></label>
      <label>Crop year <input name="crop_year" type="number" value="2024"></label>
      <label>Latitude <input name="latitude" type="number" step="0.0001" value="12.2958" required></label>
      <label>Longitude <input name="longitude" type="number" step="0.0001" value="76.6394" required></label>
      <button type="submit">Predict yield</button>
    </form>
    <pre id="result">Ready.</pre>
  </main>
  <script>
    const form = document.querySelector("#prediction-form");
    const result = document.querySelector("#result");
    const numericFields = new Set(["area_ha", "crop_year", "latitude", "longitude"]);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      result.textContent = "Predicting...";

      const payload = {};
      for (const [key, value] of new FormData(form).entries()) {
        if (value === "") continue;
        payload[key] = numericFields.has(key) ? Number(value) : value;
      }

      try {
        const response = await fetch("/predict/live", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        result.textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        result.textContent = String(error);
      }
    });
  </script>
</body>
</html>
"""


class LivePredictionRequest(BaseModel):
    crop: str = Field(..., description="Crop name, for example Rice or Wheat.")
    state: str | None = None
    district: str | None = None
    season: str | None = None
    area_ha: float | None = None
    crop_year: int | None = None
    latitude: float
    longitude: float


class LivePredictionResponse(BaseModel):
    predicted_yield_tonnes_per_hectare: float
    features_used: dict[str, Any]


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HOME_PAGE


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/live", response_model=LivePredictionResponse)
def predict(request: LivePredictionRequest) -> LivePredictionResponse:
    settings = get_settings()
    try:
        prediction = predict_live(settings, request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="Train the model before running live predictions.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return LivePredictionResponse(**prediction)
