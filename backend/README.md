# Backend (Python API and models)

Install from the repository root (recommended):

```bash
.venv/bin/pip install -e "backend/[api]"
NCAA_MODELS_DIR=models .venv/bin/uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Or from this directory:

```bash
pip install -e ".[api]"
NCAA_MODELS_DIR=../models uvicorn api.main:app --reload
```

Train artifacts default to `../models/` and training CSV to `../NCAA_Seed_Training_Set2.0.csv` (repo root). See the root [README.md](../README.md) for the full project.

Docker:

- **Full stack (API + UI):** from the **repository root**, run `docker compose up --build`. UI: [http://localhost:8080](http://localhost:8080), API: [http://localhost:8000](http://localhost:8000).
- **API only:** from this directory, `docker compose -f docker-compose.yml up --build` (see [docker-compose.yml](docker-compose.yml)).
