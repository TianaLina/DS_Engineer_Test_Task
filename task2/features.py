import pandas as pd

TARGET = "target"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from raw columns.

    EDA showed target ≈ col6**2 + col7, and the other columns are noise.
    """
    out = pd.DataFrame(index=df.index)
    out["col6_squared"] = df["6"] ** 2
    out["col7"] = df["7"]
    return out
