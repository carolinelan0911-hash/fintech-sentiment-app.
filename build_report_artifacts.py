# build_report_artifacts.py
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT/"result"; RES.mkdir(exist_ok=True)

def main():
    # Load returns (wide)
    ret_path = (ROOT/"data"/"return_matrix.csv")
    R = pd.read_csv(ret_path, parse_dates=["date"]).set_index("date").sort_index() if ret_path.exists() else None
    # Load sentiment long
    S_path = RES/"sentiment_index.csv"
    if not S_path.exists():
        print("[SKIP] sentiment_index.csv not found.")
        return
    S = pd.read_csv(S_path, parse_dates=["date"])
    if R is not None:
        S_wide = S.pivot(index="date", columns=S.columns[1], values="sentiment_index").reindex(R.index).fillna(0.0)
        # correlation: sentiment(t) vs return(t+1)
        corr = S_wide.corrwith(R.shift(-1)).rename("sentiment_corr_nextwk").to_frame()
        corr.to_csv(RES/"sentiment_corr_nextwk.csv")
        # heatmap figure
        fig, ax = plt.subplots(figsize=(10,5))
        im = ax.imshow(S_wide.fillna(0.0).T, aspect="auto", interpolation="nearest")
        ax.set_title("Sentiment Heatmap (symbol × time)")
        ax.set_yticks(range(len(S_wide.columns))); ax.set_yticklabels(S_wide.columns)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout(); fig.savefig(RES/"fig_sentiment_heatmap.png", dpi=150); plt.close(fig)
        print("💾 wrote correlation & heatmap.")
    # Copy canonical names if needed
    src = RES/"backtest_baseline_vs_sentiment.csv"
    if src.exists(): (RES/"backtest_results.csv").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    src2 = RES/"perf_summary_sentiment.csv"
    if src2.exists(): (RES/"perf_summary.csv").write_text(src2.read_text(encoding="utf-8"), encoding="utf-8")

if __name__ == "__main__":
    main()
