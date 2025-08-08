"""
sentiment_pipeline.py
Helper functions for sentiment analysis pipeline (Stage 2-4).
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Build directory structure
# -----------------------------
def build_week_dirs(base_dir="."):
    base_path = Path(base_dir).resolve()
    data_dir = base_path / "results" / "news_clean_data"
    fig_dir = base_path / "results" / "news_figs"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"base_dir": base_path, "data_dir": data_dir, "fig_dir": fig_dir}

# -----------------------------
# Stage 2: Add sentiment columns
# -----------------------------
def stage2_add_columns(raw_df):
    df = raw_df.copy()
    df['text_length'] = df['body'].astype(str).apply(len) if 'body' in df.columns else 0
    df['word_count'] = df['body'].astype(str).apply(lambda x: len(x.split())) if 'body' in df.columns else 0
    # Placeholder: add a fake sentiment score for demo
    df['compound'] = 0.05  # Default neutral score
    return df

# -----------------------------
# Stage 3: Sentiment analysis & plots
# -----------------------------
def stage3_sentiment_and_plots(df, dirs, thr=0.05):
    plt.figure(figsize=(8, 4))
    df['compound'].plot(title='Sentiment Compound Scores')
    plt.axhline(thr, color='green', linestyle='--')
    plt.axhline(-thr, color='red', linestyle='--')
    plot_path = dirs['fig_dir'] / "sentiment_plot.png"
    plt.savefig(plot_path)
    plt.close()
    return df

# -----------------------------
# Stage 4: Confusion matrix (placeholder)
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt

def stage4_confusion(df, dirs):
    cm_path = dirs['fig_dir'] / "confusion_matrix.png"
    # Placeholder matrix
    matrix = np.array([[50, 10], [8, 32]])  # 仅示例
    plt.figure(figsize=(4, 3))
    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(matrix.shape[0])
    plt.xticks(tick_marks, ['Neg', 'Pos'])
    plt.yticks(tick_marks, ['Neg', 'Pos'])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    return True

