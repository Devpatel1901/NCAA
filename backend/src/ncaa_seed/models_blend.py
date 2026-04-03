"""5-model full-data ensemble + 2 tournament-only models + blend."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

FULL_MODEL_NAMES = ("HGBR1", "HGBR2", "HGBR3", "GBR", "ET")
TOURNEY_MODEL_NAMES = ("t_HGBR", "t_ET")


def make_full_models(random_state_base: int = 42) -> dict:
    return {
        "HGBR1": HistGradientBoostingRegressor(
            max_iter=5000,
            learning_rate=0.01,
            max_depth=8,
            min_samples_leaf=1,
            l2_regularization=0.1,
            random_state=random_state_base,
        ),
        "HGBR2": HistGradientBoostingRegressor(
            max_iter=5000,
            learning_rate=0.02,
            max_depth=6,
            min_samples_leaf=2,
            l2_regularization=0.5,
            random_state=123,
        ),
        "HGBR3": HistGradientBoostingRegressor(
            max_iter=5000,
            learning_rate=0.015,
            max_depth=10,
            min_samples_leaf=1,
            l2_regularization=0.2,
            random_state=456,
        ),
        "GBR": GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=6,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=random_state_base,
        ),
        "ET": ExtraTreesRegressor(
            n_estimators=3000,
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state_base,
        ),
    }


def make_tourney_models(random_state_base: int = 42) -> dict:
    return {
        "t_HGBR": HistGradientBoostingRegressor(
            max_iter=8000,
            learning_rate=0.005,
            max_depth=10,
            min_samples_leaf=1,
            l2_regularization=0.05,
            random_state=random_state_base,
        ),
        "t_ET": ExtraTreesRegressor(
            n_estimators=5000,
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state_base,
        ),
    }


def fit_full_models(models: dict, X, y) -> None:
    for name, m in models.items():
        m.fit(X, y)


def fit_tourney_models(models: dict, X, y) -> None:
    for name, m in models.items():
        m.fit(X, y)


def predict_full_mean(models: dict, X) -> np.ndarray:
    preds = [models[name].predict(X) for name in FULL_MODEL_NAMES if name in models]
    return np.mean(preds, axis=0)


def predict_tourney_mean(models: dict, X) -> np.ndarray:
    preds = [models[name].predict(X) for name in TOURNEY_MODEL_NAMES if name in models]
    return np.mean(preds, axis=0)


def blend_full_tourney(full_scores: np.ndarray, tourney_scores: np.ndarray, tourney_weight: float) -> np.ndarray:
    return (1.0 - tourney_weight) * full_scores + tourney_weight * tourney_scores
