"""FastAPI service: /predict, /health, simple CSV upload UI."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from ncaa_seed.data_cleaning import prepare_train_targets
from ncaa_seed.data_cleaning import clean_wl_columns
from ncaa_seed.inference_pipeline import InferenceBundle

MODELS_DIR = Path(os.environ.get("NCAA_MODELS_DIR", Path(__file__).resolve().parents[3] / "models"))

_bundle: InferenceBundle | None = None


def get_bundle() -> InferenceBundle:
    global _bundle
    if _bundle is None:
        if not (MODELS_DIR / "bundle.joblib").exists():
            raise HTTPException(
                status_code=503,
                detail=f"Models not found under {MODELS_DIR}. Run scripts/train_models.py first.",
            )
        _bundle = InferenceBundle(MODELS_DIR)
    return _bundle


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if (MODELS_DIR / "bundle.joblib").exists():
        try:
            get_bundle()
        except Exception:  # noqa: BLE001
            pass
    yield
    global _bundle
    _bundle = None


def _cors_origins() -> list[str]:
    raw = os.environ.get(
        "NCAA_CORS_ORIGINS",
        "http://localhost,http://127.0.0.1,"
        "http://localhost:8080,http://127.0.0.1:8080,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="NCAA Seed Prediction", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _prediction_df_to_json_rows(out: pd.DataFrame) -> list[dict]:
    """NaN/inf-safe records for JSON."""
    safe = out.replace([np.inf, -np.inf], np.nan)
    return safe.astype(object).where(pd.notna(safe), None).to_dict(orient="records")


async def run_predict_to_dataframe(
    file: UploadFile,
    scenario: str,
    train_file: UploadFile | None,
    *,
    net_weight: float,
    model_weight: float,
) -> pd.DataFrame:
    """Core prediction path; returns the same columns as InferenceBundle.predict."""
    if scenario not in ("historical_test", "future_season"):
        raise HTTPException(400, "invalid scenario")
    bundle = get_bundle()
    content = await file.read()
    raw = pd.read_csv(BytesIO(content))
    train_ref = None
    if scenario == "historical_test":
        if train_file is None:
            raise HTTPException(400, "historical_test requires train_file upload")
        tbytes = await train_file.read()
        tr = pd.read_csv(BytesIO(tbytes))
        tr = prepare_train_targets(tr.copy())
        tr = clean_wl_columns(tr)
        train_ref = tr[["Season", "Bid Type", "Overall Seed"]].copy()
    return bundle.predict(
        raw,
        scenario=scenario,  # type: ignore[arg-type]
        train_reference=train_ref,
        net_weight=net_weight,
        model_weight=model_weight,
    )


async def run_predict_csv_body(
    file: UploadFile,
    scenario: str,
    train_file: UploadFile | None,
    *,
    net_weight: float,
    model_weight: float,
) -> Response:
    """CSV download response (internal / legacy routes)."""
    out = await run_predict_to_dataframe(
        file, scenario, train_file, net_weight=net_weight, model_weight=model_weight
    )
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )


@app.get("/health")
def health() -> dict:
    feat_path = MODELS_DIR / "feature_config.json"
    n_feat = None
    if feat_path.exists():
        n_feat = len(json.loads(feat_path.read_text())["feature_cols"])
    return {
        "status": "ok",
        "models_dir": str(MODELS_DIR),
        "bundle_present": (MODELS_DIR / "bundle.joblib").exists(),
        "n_features": n_feat,
    }


@app.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(...),
    scenario: Annotated[str, Form()] = "future_season",
    train_file: UploadFile | None = File(None),
    net_weight: Annotated[float, Form()] = 0.7,
    model_weight: Annotated[float, Form()] = 0.3,
) -> Response:
    return await run_predict_csv_body(
        file,
        scenario,
        train_file,
        net_weight=net_weight,
        model_weight=model_weight,
    )


@app.post("/api/predict")
async def api_predict_json(
    file: UploadFile = File(...),
    scenario: Annotated[str, Form()] = "future_season",
    train_file: UploadFile | None = File(None),
    net_weight: Annotated[float, Form()] = 0.7,
    model_weight: Annotated[float, Form()] = 0.3,
) -> JSONResponse:
    """JSON predictions for SPA / presentation UI."""
    out = await run_predict_to_dataframe(
        file,
        scenario,
        train_file,
        net_weight=net_weight,
        model_weight=model_weight,
    )
    return JSONResponse(
        content={
            "scenario": scenario,
            "rows": _prediction_df_to_json_rows(out),
            "n_rows": len(out),
        }
    )


_UI_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>NCAA Seed Predict</title></head>
<body>
<h1>NCAA seed prediction</h1>
<form action="/ui/predict" method="post" enctype="multipart/form-data">
  <p><label>Scenario<br/>
    <select name="scenario">
      <option value="future_season">future_season (no Bid Type / full field)</option>
      <option value="historical_test">historical_test (needs train CSV)</option>
    </select>
  </label></p>
  <p><label>Input CSV<br/><input type="file" name="file" accept=".csv" required></label></p>
  <p><label>Train CSV (historical only)<br/><input type="file" name="train_file" accept=".csv"></label></p>
  <p><button type="submit">Predict &amp; download CSV</button></p>
</form>
<p><a href="/docs">API docs</a> · <a href="/health">health</a></p>
</body>
</html>
"""


@app.get("/ui", response_class=HTMLResponse)
def ui_form() -> str:
    return _UI_HTML


@app.post("/ui/predict")
async def ui_predict(
    file: UploadFile = File(...),
    scenario: Annotated[str, Form()] = "future_season",
    train_file: UploadFile | None = File(None),
) -> Response:
    return await run_predict_csv_body(
        file,
        scenario,
        train_file,
        net_weight=0.7,
        model_weight=0.3,
    )
