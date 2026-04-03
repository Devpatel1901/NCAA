"""Stage 1–3 feature engineering (104 model features after pipeline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ncaa_seed.data_cleaning import clean_wl_columns, prepare_train_targets

DROP_COLS = [
    "RecordID",
    "Season",
    "Team",
    "Conference",
    "Overall Seed",
    "Bid Type",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NETNonConfSOS"] = df["NETNonConfSOS"].fillna(df["NETNonConfSOS"].median())
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    total_games = df["Total_W"] + df["Total_L"]
    df["Win_Pct"] = df["Total_W"] / total_games.replace(0, 1)
    df["Conf_Win_Pct"] = df["Conf_W"] / (df["Conf_W"] + df["Conf_L"]).replace(0, 1)
    df["NonConf_Win_Pct"] = df["NonConf_W"] / (df["NonConf_W"] + df["NonConf_L"]).replace(0, 1)
    df["Road_Win_Pct"] = df["Road_W"] / (df["Road_W"] + df["Road_L"]).replace(0, 1)
    for q in (1, 2, 3, 4):
        w, l = f"Q{q}_W", f"Q{q}_L"
        games = df[w] + df[l]
        df[f"Q{q}_Win_Pct"] = df[w] / games.replace(0, 1)
        df[f"Q{q}_Games"] = games
    df["Q1Q2_Wins"] = df["Q1_W"] + df["Q2_W"]
    df["Q1Q2_Losses"] = df["Q1_L"] + df["Q2_L"]
    df["Q1Q2_Win_Pct"] = df["Q1Q2_Wins"] / (df["Q1Q2_Wins"] + df["Q1Q2_Losses"]).replace(0, 1)
    df["Bad_Losses"] = df["Q3_L"] + df["Q4_L"]
    df["Has_Bad_Loss"] = (df["Bad_Losses"] > 0).astype(int)
    df["Quality_Score"] = df["Q1_W"] * 4 + df["Q2_W"] * 3 - df["Q3_L"] * 2 - df["Q4_L"] * 4
    df["Resume_Score"] = (
        df["Q1_W"] * 5
        + df["Q2_W"] * 3
        + df["Q3_W"] * 1
        + df["Q4_W"] * 0.5
        - df["Q1_L"] * 0.5
        - df["Q2_L"] * 1
        - df["Q3_L"] * 3
        - df["Q4_L"] * 5
    )
    df["NET_Change"] = df["NET Rank"] - df["PrevNET"]
    df["NET_Improved"] = (df["NET_Change"] < 0).astype(int)
    df["NET_AvgOpp_Diff"] = df["NET Rank"] - df["AvgOppNETRank"]
    df["NET_vs_SOS"] = df["NET Rank"] - df["NETSOS"]
    df["SOS_Diff"] = df["NETSOS"] - df["NETNonConfSOS"]
    df["NET_Top25"] = (df["NET Rank"] <= 25).astype(int)
    df["NET_Top50"] = (df["NET Rank"] <= 50).astype(int)
    df["NET_Top100"] = (df["NET Rank"] <= 100).astype(int)
    df["Log_NET"] = np.log1p(df["NET Rank"])
    df["Log_NETSOS"] = np.log1p(df["NETSOS"])
    df["NET_Squared"] = df["NET Rank"] ** 2
    df["NET_Cubed"] = df["NET Rank"] ** 3
    df["Is_AQ"] = (df["Bid Type"] == "AQ").astype(int)
    df["Is_AL"] = (df["Bid Type"] == "AL").astype(int)
    df["Is_Tournament"] = df["Bid Type"].notna().astype(int)
    df["Total_Games"] = total_games
    df["Road_vs_Overall"] = df["Road_Win_Pct"] - df["Win_Pct"]
    df["SOS_x_WinPct"] = df["NETSOS"] * df["Win_Pct"]
    df["NET_x_SOS"] = df["NET Rank"] * df["NETSOS"]
    return df


def build_conf_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Conference aggregates from training (Season, Conference) — notebook Stage 2."""
    return (
        train_df.groupby(["Season", "Conference"])
        .agg(
            Conf_Avg_NET=("NET Rank", "mean"),
            Conf_Med_NET=("NET Rank", "median"),
            Conf_Min_NET=("NET Rank", "min"),
            Conf_Std_NET=("NET Rank", "std"),
            Conf_Avg_WinPct=("Win_Pct", "mean"),
            Conf_Teams=("Team", "count"),
            Conf_Tourney_Teams=("Is_Tournament", "sum"),
            Conf_Avg_SOS=("NETSOS", "mean"),
            Conf_Avg_Q1W=("Q1_W", "mean"),
        )
        .reset_index()
    )


def _fill_missing_conference_columns(df: pd.DataFrame) -> None:
    """In-place fill for rows where (Season, Conference) was not in training stats."""
    for col in [
        "Conf_Avg_NET",
        "Conf_Med_NET",
        "Conf_Min_NET",
        "Conf_Std_NET",
        "Conf_Avg_WinPct",
        "Conf_Teams",
        "Conf_Tourney_Teams",
        "Conf_Avg_SOS",
        "Conf_Avg_Q1W",
    ]:
        if col not in df.columns or not df[col].isna().any():
            continue
        if col == "Conf_Avg_NET":
            conf_means = df.groupby("Conference")["NET Rank"].mean()
            df[col] = df[col].fillna(df["Conference"].map(conf_means))
        elif col == "Conf_Med_NET":
            conf_meds = df.groupby("Conference")["NET Rank"].median()
            df[col] = df[col].fillna(df["Conference"].map(conf_meds))
        elif col == "Conf_Min_NET":
            conf_mins = df.groupby("Conference")["NET Rank"].min()
            df[col] = df[col].fillna(df["Conference"].map(conf_mins))
        elif col == "Conf_Teams":
            conf_counts = df.groupby("Conference")["Team"].count()
            df[col] = df[col].fillna(df["Conference"].map(conf_counts))
        elif col == "Conf_Avg_SOS":
            conf_sos = df.groupby("Conference")["NETSOS"].mean()
            df[col] = df[col].fillna(df["Conference"].map(conf_sos))
        elif col == "Conf_Avg_Q1W":
            conf_q1 = df.groupby("Conference")["Q1_W"].mean()
            df[col] = df[col].fillna(df["Conference"].map(conf_q1))
        elif col == "Conf_Tourney_Teams":
            df[col] = df[col].fillna(3)
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)


def merge_conference_block(
    df: pd.DataFrame,
    conf_stats: pd.DataFrame,
    train_medians: pd.Series | None,
    future_style_fill: bool,
) -> pd.DataFrame:
    """Merge conference stats, add relative features; fill NaNs like train or 2026 notebook."""
    df = df.copy()
    df = df.merge(conf_stats, on=["Season", "Conference"], how="left")
    df["NET_vs_Conf_Avg"] = df["NET Rank"] - df["Conf_Avg_NET"]
    df["NET_vs_Conf_Best"] = df["NET Rank"] - df["Conf_Min_NET"]
    df["Conf_Tourney_Rate"] = df["Conf_Tourney_Teams"] / df["Conf_Teams"].replace(0, 1)

    if train_medians is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isna().any() and col in train_medians.index:
                df[col] = df[col].fillna(train_medians[col])

    if future_style_fill:
        _fill_missing_conference_columns(df)

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    return df


def add_committee_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AQ_Conf_Penalty"] = df["Is_AQ"] * df["Conf_Avg_NET"]
    df["AQ_Conf_Med_Penalty"] = df["Is_AQ"] * df["Conf_Med_NET"]
    df["AL_Conf_Strength"] = df["Is_AL"] * (400 - df["Conf_Avg_NET"])
    df["Is_Conf_Best"] = (df["NET_vs_Conf_Best"] == 0).astype(int)
    df["NET_x_AQ"] = df["NET Rank"] * df["Is_AQ"]
    df["NET_x_AL"] = df["NET Rank"] * df["Is_AL"]
    df["Conf_Rank_Approx"] = df["NET_vs_Conf_Best"]
    df["Resume_x_Conf"] = df["Resume_Score"] * (1 / (df["Conf_Avg_NET"] + 1))
    df["Q1W_vs_Conf"] = df["Q1_W"] - df["Conf_Avg_Q1W"]
    return df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Predicted_Bracket_Seed"] = np.ceil(df["NET Rank"] / 4).clip(1, 17)
    df["AQ_Penalty_Strength"] = df["Is_AQ"] * np.log1p(df["Conf_Avg_NET"])
    df["AL_Power_Conf"] = df["Is_AL"] * (df["Conf_Avg_NET"] < 100).astype(int)
    df["Q1_Win_Rate_Overall"] = df["Q1_W"] / df["Total_Games"].replace(0, 1)
    df["Q1Q2_Win_Rate_Overall"] = df["Q1Q2_Wins"] / df["Total_Games"].replace(0, 1)
    df["Bad_Loss_Rate"] = df["Bad_Losses"] / df["Total_L"].replace(0, 1)
    df["Q1_Loss_Rate"] = df["Q1_L"] / df["Total_L"].replace(0, 1)
    df["NET_vs_Resume"] = df["NET Rank"] - df["Resume_Score"].rank(ascending=False)
    df["NET_vs_Quality"] = df["NET Rank"] - df["Quality_Score"].rank(ascending=False)
    df["Conf_Depth_Score"] = df["NET_vs_Conf_Avg"] * df["Conf_Tourney_Rate"]
    df["NonConf_SOS_Quality"] = df["NETNonConfSOS"] * df["NonConf_Win_Pct"]
    df["Road_Q1Q2_proxy"] = df["Road_W"] * df["Win_Pct"]
    df["In_Bubble_Zone"] = ((df["NET Rank"] >= 20) & (df["NET Rank"] <= 60)).astype(int)
    df["Bubble_x_AQ"] = df["In_Bubble_Zone"] * df["Is_AQ"]
    df["Bubble_x_ConfStrength"] = df["In_Bubble_Zone"] * df["Conf_Avg_NET"]
    df["NET_Change_Abs"] = (df["NET Rank"] - df["PrevNET"]).abs()
    df["AQ_Power"] = df["Is_AQ"] * (df["Conf_Avg_NET"] < 80).astype(int)
    df["AQ_MidMajor"] = df["Is_AQ"] * ((df["Conf_Avg_NET"] >= 80) & (df["Conf_Avg_NET"] < 150)).astype(int)
    df["AQ_LowMajor"] = df["Is_AQ"] * (df["Conf_Avg_NET"] >= 150).astype(int)
    df["Strength_of_Record"] = df["Win_Pct"] * (400 - df["AvgOppNET"]) / 400
    df["Q_Balance"] = (df["Q1_W"] + df["Q2_W"]) - (df["Q3_L"] + df["Q4_L"])
    return df


def features_stage1_through_3(
    df: pd.DataFrame,
    conf_stats: pd.DataFrame,
    train_medians: pd.Series | None,
    *,
    future_style_fill: bool,
) -> pd.DataFrame:
    df = engineer_features(df)
    df = merge_conference_block(df, conf_stats, train_medians, future_style_fill=future_style_fill)
    df = add_committee_features(df)
    df = add_advanced_features(df)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(0)
    return df


def feature_column_list(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def build_training_matrix(train_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.Series]:
    """End-to-end training feature build; returns train_fe, feature_names, conf_stats, medians_after_stage1_merge."""
    train_raw = prepare_train_targets(train_raw.copy())
    train = clean_wl_columns(train_raw)
    train = engineer_features(train)
    conf_stats = build_conf_stats(train)
    train = train.merge(conf_stats, on=["Season", "Conference"], how="left")
    train["NET_vs_Conf_Avg"] = train["NET Rank"] - train["Conf_Avg_NET"]
    train["NET_vs_Conf_Best"] = train["NET Rank"] - train["Conf_Min_NET"]
    train["Conf_Tourney_Rate"] = train["Conf_Tourney_Teams"] / train["Conf_Teams"]
    for col in train.select_dtypes(include=[np.number]).columns:
        if train[col].isna().any():
            train[col] = train[col].fillna(train[col].median())
    train = add_committee_features(train)
    train = add_advanced_features(train)
    for col in train.select_dtypes(include=[np.number]).columns:
        train[col] = train[col].fillna(0)
    feat_cols = feature_column_list(train)
    train_medians_full = train.median(numeric_only=True)
    return train, feat_cols, conf_stats, train_medians_full
