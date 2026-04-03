# Presentation UI

Single-page app: model predictions (API) vs reference CSV (local + bundled).

## Setup

```bash
cd frontend
cp .env.example .env   # optional; defaults to http://127.0.0.1:8000
npm install
```

## Backend (from repo root)

```bash
.venv/bin/pip install -e "backend/[api]"
NCAA_MODELS_DIR=models .venv/bin/uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

(`api.main` lives under `backend/src/api/`.)

## Run UI

```bash
cd frontend && npm run dev
```

Open the Vite URL. The right panel preloads **`public/reference/tournament_seeds_2025-26.csv`** (internet-style S-curve list). Drop **2026** team stats on the left to call **`POST /api/predict`**.

## Docker (production build)

From the **repository root**:

```bash
docker compose up --build
```

- **UI:** [http://localhost:8080](http://localhost:8080) (nginx + static `npm run build`)
- **API:** [http://localhost:8000](http://localhost:8000)

The image bakes `VITE_API_URL=http://localhost:8000` so the browser calls the API on the host. Ensure `./models/` contains `bundle.joblib` before starting.

Backend-only: see [backend/README.md](../backend/README.md).

## Build

```bash
npm run build && npm run preview
```
