"""Offline training: fit full + tournament ensembles, persist artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ncaa_seed.features import build_training_matrix
from ncaa_seed.models_blend import (
    blend_full_tourney,
    fit_full_models,
    fit_tourney_models,
    make_full_models,
    make_tourney_models,
    predict_full_mean,
    predict_tourney_mean,
)

DEFAULT_TOURNEY_WEIGHT = 0.1


def train_and_save(
    train_csv: Path | str,
    out_dir: Path | str,
    *,
    tourney_blend_weight: float = DEFAULT_TOURNEY_WEIGHT,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(train_csv)
    train_fe, feature_cols, conf_stats, train_medians = build_training_matrix(train_raw)

    X = train_fe[feature_cols]
    y = train_fe["Overall Seed"]

    full_models = make_full_models()
    fit_full_models(full_models, X, y)

    mask_t = train_fe["Overall Seed"] > 0
    X_t = train_fe.loc[mask_t, feature_cols]
    y_t = train_fe.loc[mask_t, "Overall Seed"]
    tourney_models = make_tourney_models()
    fit_tourney_models(tourney_models, X_t, y_t)

    artifacts = {
        "full_models": full_models,
        "tourney_models": tourney_models,
        "feature_cols": feature_cols,
        "conf_stats": conf_stats,
        "train_medians": train_medians,
    }
    joblib.dump(artifacts, out_dir / "bundle.joblib")

    blend_cfg = {
        "tourney_blend_weight": tourney_blend_weight,
        "n_features": len(feature_cols),
    }
    (out_dir / "blend_config.json").write_text(json.dumps(blend_cfg, indent=2))

    feat_cfg = {"feature_cols": feature_cols}
    (out_dir / "feature_config.json").write_text(json.dumps(feat_cfg, indent=2))

    return {
        "out_dir": str(out_dir),
        "n_features": len(feature_cols),
        "blend_config": blend_cfg,
    }
