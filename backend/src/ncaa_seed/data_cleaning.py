"""Win–loss parsing (Excel date corruption) and column cleaning."""

from __future__ import annotations

import pandas as pd

MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def parse_wl(val):
    """Parse a W–L string into (wins, losses); handles month-style Excel corruption."""
    if pd.isna(val) or str(val).strip() == "":
        return 0, 0
    parts = str(val).strip().split("-")
    if len(parts) != 2:
        return 0, 0
    left, right = parts[0].strip(), parts[1].strip()
    left_num = MONTH_MAP.get(left, None)
    right_num = MONTH_MAP.get(right, None)
    if left_num is not None and right_num is not None:
        return left_num, right_num
    if left_num is not None:
        return left_num, int(right)
    if right_num is not None:
        return int(left), right_num
    return int(left), int(right)


def to_num_wl(val):
    """Alias for parse_wl — same numeric conversion used in training and inference."""
    return parse_wl(val)


def clean_wl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Split W–L-like columns into integer wins/losses columns."""
    df = df.copy()
    wl_cols = {
        "WL": ("Total_W", "Total_L"),
        "Conf.Record": ("Conf_W", "Conf_L"),
        "Non-ConferenceRecord": ("NonConf_W", "NonConf_L"),
        "RoadWL": ("Road_W", "Road_L"),
        "Quadrant1": ("Q1_W", "Q1_L"),
        "Quadrant2": ("Q2_W", "Q2_L"),
        "Quadrant3": ("Q3_W", "Q3_L"),
        "Quadrant4": ("Q4_W", "Q4_L"),
    }
    for col, (w_name, l_name) in wl_cols.items():
        if col in df.columns:
            parsed = df[col].apply(parse_wl)
            df[w_name] = parsed.apply(lambda x: x[0])
            df[l_name] = parsed.apply(lambda x: x[1])
            df.drop(columns=[col], inplace=True)
    return df


def prepare_train_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Non-tournament rows → Overall Seed 0 (training only)."""
    df = df.copy()
    df.loc[df["Bid Type"].isna(), "Overall Seed"] = 0
    df["Overall Seed"] = df["Overall Seed"].fillna(0)
    return df
