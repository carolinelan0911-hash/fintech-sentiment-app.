"""
merge_data.py
Merge structured crypto data (stage_2_crypto_data.csv) with unstructured news sentiment data (clean_news_timeseries.csv)
to create a final dataset for portfolio optimization and sentiment analysis.
"""

import pandas as pd
from pathlib import Path

# ------------------------- User-defined paths -------------------------
STRUCTURED_FILE = Path("week_crypto/results/clean_data/stage_2_crypto_data.csv")
UNSTRUCTURED_FILE = Path("results/news_clean_data/clean_news_timeseries.csv")
OUTPUT_FILE = Path("results/final_dataset.csv")
# ----------------------------------------------------------------------

def main():
    # 1. Load structured data
    print(f"Loading structured data from {STRUCTURED_FILE} ...")
    structured = pd.read_csv(STRUCTURED_FILE, parse_dates=["date"])
    print(f"Structured data shape: {structured.shape}")

    # 2. Load unstructured news sentiment data
    print(f"Loading unstructured sentiment data from {UNSTRUCTURED_FILE} ...")
    unstructured = pd.read_csv(UNSTRUCTURED_FILE, parse_dates=["date"])
    print(f"Unstructured data shape: {unstructured.shape}")

    # 3. Aggregate news sentiment by date
    # (Average sentiment scores per day, or sum of positive/negative counts)
    sentiment_agg = (
        unstructured.groupby("date")
        .agg(
            avg_sentiment=("compound", "mean"),
            news_count=("compound", "count")
        )
        .reset_index()
    )
    print("Aggregated sentiment data:")
    print(sentiment_agg.head())

    # 4. Merge structured data with sentiment data
    final_df = pd.merge(structured, sentiment_agg, on="date", how="left")

    # 5. Fill missing sentiment values (no news days)
    final_df["avg_sentiment"].fillna(0, inplace=True)
    final_df["news_count"].fillna(0, inplace=True)

    # 6. Save final dataset
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Final dataset saved to {OUTPUT_FILE}")
    print(f"Final dataset shape: {final_df.shape}")

if __name__ == "__main__":
    main()
