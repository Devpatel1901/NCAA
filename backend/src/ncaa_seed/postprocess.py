"""Historical constrained assignment vs future-season field construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def constrained_assignment(
    ensemble_pred: np.ndarray | pd.Series,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.Series:
    """Kaggle-style: assign missing seeds 1–68 to test rows with Bid Type; others 0."""
    if isinstance(ensemble_pred, pd.Series):
        arr = ensemble_pred.reindex(test_df.index).values
    else:
        arr = ensemble_pred
    final = pd.Series(0, index=test_df.index, dtype=int)
    for season in sorted(test_df["Season"].unique()):
        tr_seeds = (
            train_df[(train_df["Season"] == season) & (train_df["Bid Type"].notna())]["Overall Seed"]
            .astype(int)
            .tolist()
        )
        missing_seeds = sorted(set(range(1, 69)) - set(tr_seeds))
        mask = (test_df["Season"] == season) & (test_df["Bid Type"].notna())
        test_idx = test_df[mask].index
        team_scores = pd.Series(arr[test_df.index.get_indexer(test_idx)], index=test_idx).sort_values()
        for idx, seed in zip(team_scores.index, missing_seeds):
            final[idx] = int(seed)
    return final


def future_season_field(
    df_feat: pd.DataFrame,
    blended_model_scores: np.ndarray,
    *,
    net_weight: float = 0.7,
    model_weight: float = 0.3,
    enforce_conference_autobids: bool = True,
) -> pd.Series:
    """
    Unseen season: NET + blend hybrid, top 68, optional conference auto-bids (notebook logic).
    Lower blended score = stronger (selected earlier for nsmallest 68).
    """
    net_scores = df_feat["NET Rank"].values.astype(float)
    hybrid = model_weight * blended_model_scores + net_weight * net_scores
    top_68_idx = pd.Series(hybrid, index=df_feat.index).nsmallest(68).sort_values().index

    if not enforce_conference_autobids:
        seeds = pd.Series(0, index=df_feat.index, dtype=int)
        for rank, idx in enumerate(top_68_idx, 1):
            seeds[idx] = rank
        return seeds

    bracket_teams = df_feat.loc[top_68_idx]
    conf_counts = bracket_teams["Conference"].value_counts()
    all_confs = df_feat["Conference"].unique()
    missing_confs = [c for c in all_confs if c not in conf_counts.index]

    auto_bids: dict = {}
    for conf in missing_confs:
        pool = df_feat[df_feat["Conference"] == conf]
        if len(pool) == 0:
            continue
        best = pool.nsmallest(1, "NET Rank")
        if len(best) > 0:
            auto_bids[best.index[0]] = {"team": best["Team"].values[0], "conf": conf}

    current_top68 = list(top_68_idx)
    n_remove = len(auto_bids)
    removable: list = []
    for idx in reversed(current_top68):
        conf = df_feat.loc[idx, "Conference"]
        conf_in_bracket = sum(1 for i in current_top68 if df_feat.loc[i, "Conference"] == conf)
        if conf_in_bracket > 1:
            removable.append(idx)
        if len(removable) >= n_remove:
            break

    final_68 = [idx for idx in current_top68 if idx not in removable]
    final_68.extend(auto_bids.keys())

    final_scores = pd.Series(hybrid[df_feat.index.get_indexer(final_68)], index=final_68).sort_values()
    seeds = pd.Series(0, index=df_feat.index, dtype=int)
    for rank, idx in enumerate(final_scores.index, 1):
        seeds[idx] = rank
    return seeds
