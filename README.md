# Flat Price Analyzer (Poland)

Predicts **monthly rent** and **sale price** of an apartment in 15 Polish cities from its parameters.
React + Vite frontend → FastAPI backend → XGBoost models (one for rent, one for sale).

```
frontend/   React SPA (form, Leaflet map pin, results)
backend/    FastAPI: /api/meta, /api/predict, /api/health (+ serves the SPA in prod)
flatprice/  shared feature encoding used by training AND serving (no train/serve skew)
ml/         training: prepare.py, train.py, sagemaker_train.py (optional lab flow)
artifacts/  trained models: {rent,sale}/model.ubj, columns.json, metrics.json, meta.json
```

## Current model quality (held-out test set, 15%)

| Target | MAE | RMSE | R² | Median-baseline MAE |
|---|---|---|---|---|
| Rent (PLN/month) | ~434 | ~677 | 0.87 | ~1217 |
| Sale (PLN) | ~71.7k | ~111k | 0.91 | ~261k |

Trained on all cities with `city` as a feature (the lab's "model B"); per-city MAE is in `metrics.json`
and is used for the ± range shown in the UI.

## Data

Kaggle "Apartment Prices in Poland" (`krzysztofjamroz/apartment-prices-in-poland`). Put the CSVs in `ml/data/`:

```bash
curl -L -o /tmp/ds.zip https://www.kaggle.com/api/v1/datasets/download/krzysztofjamroz/apartment-prices-in-poland && unzip -o /tmp/ds.zip -d ml/data
```

Listings repeat month to month, so `prepare.py` keeps only the latest snapshot per `id` before the 70/15/15 split (otherwise copies of the same flat leak into the test set).

## Local setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
```
(macOS: XGBoost needs `brew install libomp`.)

Train (writes `artifacts/<target>/`):
```bash
.venv/bin/python ml/train.py --target rent
.venv/bin/python ml/train.py --target sale
```

Run the API and the frontend dev server:
```bash
cd backend && PYTHONPATH=.. ../.venv/bin/uvicorn app.main:app --reload --port 8000
```
```bash
cd frontend && npm install && npm run dev
```
Open http://localhost:5173 (Vite proxies `/api` to :8000).

Tests:
```bash
cd backend && ../.venv/bin/python -m pytest -q
```

## API

`POST /api/predict`
```json
{ "city": "czestochowa", "type": "blockOfFlats", "squareMeters": 52, "rooms": 2,
  "floor": 3, "floorCount": 5, "buildYear": 1995, "latitude": 50.81, "longitude": 19.12,
  "hasBalcony": true, "targets": ["rent", "sale"], "fillMissing": true }
```
Only `city`, `squareMeters`, `rooms` are required. With `fillMissing` (default), `centreDistance` is computed from the
coordinates and the other distance fields fall back to city medians; anything else missing stays NaN (XGBoost handles it).
Response: `{ rent: {price, pricePerM2, low, high, mae}, sale: {...}, imputed: {...}, predictor }`.

`GET /api/meta` – allowed categorical values, city centres/medians, model metrics.

## Docker / hosting

Local dev, everything with hot reload (UI http://localhost:5173, API docs http://localhost:8000/docs):
```bash
docker compose up --build
```
```bash
docker compose down
```
Production-like single image (API + built SPA + models, what Render runs) on http://localhost:8000:
```bash
docker compose -f docker-compose.prod.yml up --build
```

**Render (free tier):** push the repo to GitHub → Render → New → Blueprint → pick the repo (`render.yaml`).
The free tier sleeps after inactivity (first request ~30–60 s). Any Docker host (Fly.io, Railway, AWS App Runner, a VPS) works the same way; the app listens on `$PORT`.

## Using the SageMaker endpoint instead (lab flow)

```bash
.venv/bin/python ml/train.py --target rent --write-sagemaker-csv   # target-first, no header CSVs
pip install sagemaker && python ml/sagemaker_train.py --target rent  # Training Job + Serverless endpoint
```
Then run the backend with `PREDICTOR=sagemaker SAGEMAKER_RENT_ENDPOINT=<name> SAGEMAKER_SALE_ENDPOINT=<name>` and AWS
credentials of an IAM user that only has `sagemaker:InvokeEndpoint`. The backend sends the same `columns.json`-ordered CSV row
as the lab's Node.js example. Delete endpoints when done: `python ml/sagemaker_train.py --delete <name>`.
