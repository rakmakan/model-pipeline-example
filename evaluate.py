"""
Evaluates a candidate model on its test set and the retrospective replay dataset.

Usage:
    python evaluate.py --model-version v1 --replay-version v1
"""
import argparse
from model_pipeline.evaluator import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--replay-version", required=True)
    args = parser.parse_args()
    run(args.model_version, args.replay_version)
