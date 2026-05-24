"""
Finds the optimal operating threshold for a candidate model on the validation set.

Usage:
    python validate.py --model-version v1
"""
import argparse

from logging_config import setup_logging
from model_pipeline.validator import run

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()
    run(args.model_version)
