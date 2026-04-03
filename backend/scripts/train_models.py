#!/usr/bin/env python3
"""Train and save blend ensemble artifacts (default: repo ../models)."""

import argparse
from pathlib import Path

from ncaa_seed.training_pipeline import train_and_save

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Train NCAA seed blend models")
    p.add_argument(
        "--train-csv",
        type=Path,
        default=_REPO_ROOT / "NCAA_Seed_Training_Set2.0.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "models",
    )
    p.add_argument("--tourney-weight", type=float, default=0.1)
    args = p.parse_args()
    info = train_and_save(args.train_csv, args.out_dir, tourney_blend_weight=args.tourney_weight)
    print(info)


if __name__ == "__main__":
    main()
