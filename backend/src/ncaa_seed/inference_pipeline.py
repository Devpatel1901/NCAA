"""Load artifacts and run historical_test vs future_season inference paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from ncaa_seed.data_cleaning import clean_wl_columns
from ncaa_seed.features import features_stage1_through_3
from ncaa_seed.models_blend import blend_full_tourney, predict_full_mean, predict_tourney_mean
from ncaa_seed.postprocess import constrained_assignment, future_season_field

Scenario = Literal["historical_test", "future_season"]


class InferenceBundle:
    def __init__(self, models_dir: Path | str):
        models_dir = Path(models_dir)
        self.models_dir = models_dir
        blob = joblib.load(models_dir / "bundle.joblib")
        self.full_models = blob["full_models"]
        self.tourney_models = blob["tourney_models"]
        self.feature_cols: list[str] = blob["feature_cols"]
        self.conf_stats: pd.DataFrame = blob["conf_stats"]
        self.train_medians: pd.Series = blob["train_medians"]
        blend_path = models_dir / "blend_config.json"
        self.tourney_blend_weight = 0.1
        if blend_path.exists():
            self.tourney_blend_weight = float(json.loads(blend_path.read_text())["tourney_blend_weight"])

    def featurize(
        self,
        raw: pd.DataFrame,
        *,
        future_style_fill: bool,
    ) -> pd.DataFrame:
        df = clean_wl_columns(raw.copy())
        df = features_stage1_through_3(
            df,
            self.conf_stats,
            self.train_medians,
            future_style_fill=future_style_fill,
        )
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        return df

    def predict_blended_scores(self, X: pd.DataFrame) -> np.ndarray:
        mat = X[self.feature_cols]
        full = predict_full_mean(self.full_models, mat)
        tour = predict_tourney_mean(self.tourney_models, mat)
        return blend_full_tourney(full, tour, self.tourney_blend_weight)

    def predict(
        self,
        raw: pd.DataFrame,
        *,
        scenario: Scenario,
        train_reference: pd.DataFrame | None = None,
        net_weight: float = 0.7,
        model_weight: float = 0.3,
        enforce_conference_autobids: bool = True,
    ) -> pd.DataFrame:
        if scenario == "historical_test":
            if train_reference is None:
                raise ValueError("historical_test requires train_reference (Season, Bid Type, Overall Seed).")
            feat = self.featurize(raw, future_style_fill=False)
            blended = self.predict_blended_scores(feat)
            seeds = constrained_assignment(blended, raw, train_reference)
            out = raw[["RecordID", "Season", "Team"]].copy()
            out["Predicted_Seed"] = seeds.values
            out["Blended_Model_Score"] = blended
            return out

        feat = self.featurize(raw, future_style_fill=True)
        blended_model = self.predict_blended_scores(feat)
        hybrid_final = future_season_field(
            feat,
            blended_model,
            net_weight=net_weight,
            model_weight=model_weight,
            enforce_conference_autobids=enforce_conference_autobids,
        )
        out = raw[["RecordID", "Season", "Team"]].copy()
        out["Predicted_Seed"] = hybrid_final.values
        out["Blended_Model_Score"] = blended_model
        return out


def run_inference_csv(
    models_dir: Path | str,
    input_csv: Path | str,
    output_csv: Path | str,
    *,
    scenario: Scenario,
    train_csv: Path | str | None = None,
) -> None:
    bundle = InferenceBundle(models_dir)
    raw = pd.read_csv(input_csv)
    train_ref = None
    if scenario == "historical_test":
        if train_csv is None:
            raise ValueError("--train-csv required for historical_test")
        tr = pd.read_csv(train_csv)
        from ncaa_seed.data_cleaning import prepare_train_targets

        tr = prepare_train_targets(tr.copy())
        tr = clean_wl_columns(tr)
        train_ref = tr[["Season", "Bid Type", "Overall Seed"]].copy()
    out = bundle.predict(raw, scenario=scenario, train_reference=train_ref)
    out.to_csv(output_csv, index=False)
