"""Train a regression model and save it to disk.

Usage:
    python train.py --train-path data/train.csv --model-path models/model.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from features import TARGET, make_features


def parse_args():
    p = argparse.ArgumentParser(description="Train regression model.")
    p.add_argument("--train-path", default="data/train.csv")
    p.add_argument("--model-path", default="models/model.joblib")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(make_features)),
        ("model", LinearRegression()),
    ])


def main():
    args = parse_args()
    import pandas as pd
    df = pd.read_csv(args.train_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]

    # Hold-out validation to report an honest RMSE
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed
    )
    pipe = build_pipeline().fit(X_tr, y_tr)
    rmse = np.sqrt(mean_squared_error(y_val, pipe.predict(X_val)))
    print(f"Validation RMSE: {rmse:.6f}")

    # Refit on all data for the final model
    pipe.fit(X, y)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()