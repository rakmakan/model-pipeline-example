"""
Validates, splits, and versions a raw training CSV.

Usage:
    python run_data_pipe.py --input data/raw/applications_v2.csv
    python run_data_pipe.py --input data/raw/applications_v2.csv --version v2
"""
import argparse
from data_pipeline.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--version", default=None, help="Version name (auto-increments if omitted)")
    args = parser.parse_args()
    run(args.input, args.version)
