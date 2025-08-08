"""
run_sentiment_pipeline.py
Run stages 2-4 of the CoinDesk news sentiment pipeline.
Generates cleaned news dataset (clean_news_timeseries.csv) and sentiment plots.
"""

from pathlib import Path
import matplotlib as mpl
import pandas as pd

from sentiment_pipeline import (
    build_week_dirs,
    stage2_add_columns,
    stage3_sentiment_and_plots,
    stage4_confusion,
)

# --------------------------- user settings
BASE_DIR = "."                            # Current directory as base
WHITE_BACKGROUND = True                   # Plot background color
THR = 0.05                                # |compound| threshold for sentiment
RAW_FILE = "stage_1_news_raw.csv"         # Input CSV of raw news data
OUTPUT_FILE = "clean_news_timeseries.csv" # Output cleaned news dataset
# ----------------------------------------

# Apply style settings
if WHITE_BACKGROUND:
    mpl.rcdefaults()
else:
    mpl.pyplot.style.use("dark_background")

def main() -> None:
    """
    Main pipeline execution for unstructured data:
    - Stage 2: Text preprocessing and feature columns
    - Stage 3: Sentiment analysis and visualization
    - Stage 4: Confusion matrix & performance metrics
    """
    # Prepare directories
    dirs = build_week_dirs(BASE_DIR)

    # Locate and load Stage 1 raw news data
    candidates = [Path(BASE_DIR) / RAW_FILE, dirs["data_dir"] / RAW_FILE]
    for p in candidates:
        if p.exists():
            raw_path = p
            break
    else:
        raise FileNotFoundError(f"{RAW_FILE} not found in {candidates}")

    raw = pd.read_csv(raw_path)
    print(f"Loaded {len(raw):,} raw articles from {raw_path}")

    # Stage 2: Add sentiment-related columns and text features
    clean = stage2_add_columns(raw)

    # Stage 3: Sentiment analysis and plots
    sent = stage3_sentiment_and_plots(clean, dirs, thr=THR)

    # Stage 4: Generate confusion matrix & performance metrics
    stage4_confusion(sent, dirs)

    # Save cleaned news dataset
    final_csv = dirs["data_dir"] / OUTPUT_FILE
    sent.to_csv(final_csv, index=False)
    print("Done!")
    print(f"Final CSV -> {final_csv.resolve()}")
    print(f"Plots     -> {dirs['fig_dir'].resolve()}")

if __name__ == "__main__":
    main()
