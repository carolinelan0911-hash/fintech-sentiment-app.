"""
feature_engineering.py
Generates engineered features for crypto portfolio optimization.

Inputs:
- results/final_dataset.csv (merged structured + unstructured data)

Outputs:
- results/features_dataset.csv (enhanced dataset with new factors)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
INPUT_PATH = Path("results/final_dataset.csv")
OUTPUT_PATH = Path("results/features_dataset.csv")
ROLLING_WINDOWS = [7, 14]  # example windows for rolling features
# ----------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling features and lagged returns to dataset."""
    df = df.copy()

    # Ensure date is sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Lagged return (1 day)
    df["return_lag1"] = df.groupby("symbol")["return"].shift(1)

    # Rolling volatility (7-day)
    df["volatility_7d"] = (
        df.groupby("symbol")["return"].rolling(7).std().reset_index(level=0, drop=True)
    )

    # Momentum (14-day cumulative return)
    df["momentum_14d"] = (
        df.groupby("symbol")["return"]
        .rolling(14)
        .apply(lambda x: (1 + x).prod() - 1, raw=False)
        .reset_index(level=0, drop=True)
    )

    # Rolling avg_sentiment (7-day)
    df["rolling_sentiment_7d"] = (
        df.groupby("symbol")["avg_sentiment"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # Rolling news count (7-day sum)
    df["rolling_news_count_7d"] = (
        df.groupby("symbol")["news_count"]
        .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )

    return df


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found. Please generate final_dataset.csv first.")

    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset loaded with shape {df.shape}")

    df_features = add_features(df)
    df_features.to_csv(OUTPUT_PATH, index=False)
    print(f"Features dataset saved to {OUTPUT_PATH}")
    print(f"Final shape: {df_features.shape}")
    print(df_features.head())


if __name__ == "__main__":
    main()


    def main():
        if not INPUT_PATH.exists():
            raise FileNotFoundError(f"{INPUT_PATH} not found. Please generate final_dataset.csv first.")

        print(f"Loading {INPUT_PATH} ...")
        df = pd.read_csv(INPUT_PATH)
        print(f"Dataset loaded with shape {df.shape}")

        df_features = add_features(df)
        df_features.to_csv(OUTPUT_PATH, index=False)
        print(f"Features dataset saved to {OUTPUT_PATH}")
        print(f"Final shape: {df_features.shape}")
        print(df_features.head())


        selected = df_features[df_features["symbol"].isin(["BTC", "ETH", "ADA"])]
        stats_selected = selected.groupby("symbol")["return"].agg(
            mean_return="mean",
            volatility="std",
            max_return="max",
            min_return="min",
            skew="skew",
            kurt="kurt"
        )
        stats_selected.to_csv("results/descriptive_stats_top3.csv")
        print("Descriptive stats for BTC/ETH/ADA saved to results/descriptive_stats_top3.csv")
