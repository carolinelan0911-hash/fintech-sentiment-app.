"""
analyze_features.py
Analyze and visualize engineered features from features_dataset.csv.

Outputs:
- results/features_analysis/descriptive_stats_features.csv
- results/features_analysis/correlation_matrix.csv
- results/features_analysis/feature_histograms.png
- results/features_analysis/feature_correlations.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
FEATURES_PATH = Path("results/features_dataset.csv")
OUTPUT_DIR = Path("results/features_analysis")
# ----------------------------------------

def plot_correlation_matrix(df, output_path):
    """Draw a correlation heatmap using matplotlib only."""
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)

    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    plt.title("Feature Correlation Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_histograms(df, output_path):
    """Plot histograms for numeric columns using matplotlib."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols].hist(bins=30, figsize=(15, 12))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"{FEATURES_PATH} does not exist. Please run feature_engineering.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)

    print(f"Analyzing dataset: {FEATURES_PATH}, shape: {df.shape}")

    # 1. Descriptive statistics
    stats = df.describe().T
    stats.to_csv(OUTPUT_DIR / "descriptive_stats_features.csv")
    print(f"Descriptive stats saved to {OUTPUT_DIR / 'descriptive_stats_features.csv'}")

    # 2. Correlation matrix CSV
    corr = df.corr(numeric_only=True)
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print(f"Correlation matrix saved to {OUTPUT_DIR / 'correlation_matrix.csv'}")

    # 3. Correlation heatmap
    plot_correlation_matrix(df, OUTPUT_DIR / "feature_correlations.png")
    print(f"Correlation heatmap saved to {OUTPUT_DIR / 'feature_correlations.png'}")

    # 4. Histograms
    plot_histograms(df, OUTPUT_DIR / "feature_histograms.png")
    print(f"Histograms saved to {OUTPUT_DIR / 'feature_histograms.png'}")


if __name__ == "__main__":
    analyze_features()


    def plot_sentiment_vs_price(df, output_path, crypto="BTC"):
        """Plot rolling sentiment vs price for a specific crypto."""
        subset = df[df["symbol"] == crypto].copy()
        if "rolling_sentiment_7d" not in subset.columns or "close" not in subset.columns:
            print(f"rolling_sentiment_7d or close column not found for {crypto}")
            return
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.plot(subset["date"], subset["close"], color="blue", label="Price (USD)")
        ax2.plot(subset["date"], subset["rolling_sentiment_7d"], color="orange", label="Rolling Sentiment")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (USD)", color="blue")
        ax2.set_ylabel("Rolling Sentiment", color="orange")
        plt.title(f"{crypto} Price vs Rolling Sentiment (7d)")
        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()


    def analyze_features():
        ...
        # Existing analysis
        plot_sentiment_vs_price(df, OUTPUT_DIR / "btc_sentiment_vs_price.png", crypto="BTC")

