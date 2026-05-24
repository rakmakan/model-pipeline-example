"""
Joins predictions and feedback into a versioned replay dataset for gate evaluation.

Usage:
    python run_replay_pipe.py --predictions data/predictions.csv --feedback data/feedback.csv
"""
import argparse

from logging_config import setup_logging
from replay_pipeline.pipeline import run

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    run(args.predictions, args.feedback, args.version)
