#!/usr/bin/env python3
"""CLI batch inference."""

import argparse
from pathlib import Path

from ncaa_seed.inference_pipeline import run_inference_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Batch predict NCAA seeds")
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument(
        "--scenario",
        choices=("historical_test", "future_season"),
        required=True,
    )
    p.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Training CSV (for historical_test constrained assignment)",
    )
    args = p.parse_args()
    run_inference_csv(
        args.models_dir,
        args.input_csv,
        args.output_csv,
        scenario=args.scenario,
        train_csv=args.train_csv,
    )
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
