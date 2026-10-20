"""Load a trained model and generate predictions.

Usage:
    python predict.py --input data/hidden_test.csv \
        --model-path models/model.joblib --output predictions/predictions.csv
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import features  # noqa: F401  (needed so joblib can unpickle make_features)


def parse_args():
    p = argparse.ArgumentParser(description="Predict with trained model.")
    p.add_argument("--input", default="data/hidden_test.csv")
    p.add_argument("--model-path", default="models/model.joblib")
    p.add_argument("--output", default="predictions/predictions.csv")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    pipe = joblib.load(args.model_path)

    preds = pipe.predict(df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target": preds}).to_csv(args.output, index=False)
    print(f"Saved {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()