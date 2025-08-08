# sentiment_portfolio.py
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional

def adjust_weights_with_sentiment(weights_df: pd.DataFrame,
                                  sentiment_df: pd.DataFrame,
                                  feature_col: str = "sentiment_index",
                                  scale: float = 0.25,
                                  wmin: float = 0.0,
                                  wmax: float = 0.30) -> pd.DataFrame:
    W = weights_df.copy()
    if W.index.name != "date": W.index.name = "date"
    if not isinstance(W.index, pd.DatetimeIndex):
        try: W.index = pd.to_datetime(W.index, errors="coerce")
        except Exception: pass
    W_long = W.reset_index().melt(id_vars="date", var_name="symbol", value_name="weight")

    S = sentiment_df.copy()
    if "date" not in S.columns: raise ValueError("sentiment_df must contain 'date'.")
    if "symbol" in S.columns:
        if feature_col not in S.columns:
            alt = next((c for c in ["sentiment_index","avg_sentiment","compound","vader_compound","sentiment"] if c in S.columns), None)
            if alt is None: raise ValueError("Sentiment column not found.")
            S = S.rename(columns={alt: feature_col})
        S_long = S[["date","symbol",feature_col]].copy()
    else:
        value_cols = [c for c in S.columns if str(c).lower() not in ("date","news_count")]
        if not value_cols: raise ValueError("Wide-format sentiment_df has no asset columns.")
        S_long = S.melt(id_vars="date", value_vars=value_cols, var_name="symbol", value_name=feature_col)

    S_long["date"] = pd.to_datetime(S_long["date"], errors="coerce")
    W_long["date"] = pd.to_datetime(W_long["date"], errors="coerce")

    merged = pd.merge(W_long, S_long[["date","symbol",feature_col]], on=["date","symbol"], how="left")
    merged[feature_col] = merged[feature_col].fillna(0.0)
    merged["weight_adj"] = merged["weight"] * (1.0 + scale*merged[feature_col])
    merged["weight_adj"] = np.clip(merged["weight_adj"], wmin, wmax)
    sum_w = merged.groupby("date")["weight_adj"].transform("sum")
    cnt_w = merged.groupby("date")["weight_adj"].transform("count")
    merged["weight_adj"] = np.where((sum_w<=0)|(~np.isfinite(sum_w)), 1.0/cnt_w, merged["weight_adj"]/sum_w)
    return merged.pivot(index="date", columns="symbol", values="weight_adj").sort_index()

def plot_portfolio_vs_benchmark(returns_base: pd.Series,
                                returns_sent: pd.Series,
                                bench: Optional[pd.Series],
                                out_path: str) -> str:
    eq_base=(1+returns_base.dropna()).cumprod()
    eq_sent=(1+returns_sent.dropna()).cumprod()
    fig, ax = plt.subplots(figsize=(8,4))
    if not eq_base.empty: ax.plot(eq_base.index, eq_base.values, label="Baseline")
    if not eq_sent.empty: ax.plot(eq_sent.index, eq_sent.values, label="Sentiment")
    if bench is not None and not bench.empty:
        ax.plot(bench.index, (1+bench).cumprod().values, label="Benchmark", alpha=0.8)
    ax.set_title("Cumulative Return — Baseline vs Sentiment"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path
