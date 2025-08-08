# features_dataset.py
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

def build_features(returns_wide: pd.DataFrame,
                   sentiment_long: pd.DataFrame,
                   vol_window: int = 20,
                   mom_window: int = 60,
                   senti_window: int = 7) -> pd.DataFrame:
    R = returns_wide.copy()
    if "date" in R.columns:
        R["date"] = pd.to_datetime(R["date"], errors="coerce")
        R = R.set_index("date")
    R.index = pd.to_datetime(R.index, errors="coerce")
    R = R.sort_index().dropna(how="all")
    ret_long = R.reset_index().melt(id_vars="date", var_name="symbol", value_name="return")

    def _roll_std(s): return s.rolling(vol_window, max(2,vol_window//2)).std()
    def _roll_mom(s):
        x = (1.0+s).rolling(mom_window, max(2,mom_window//2)).apply(lambda a: np.prod(a)-1.0, raw=True)
        return x

    ret_long["vol_20d"] = ret_long.groupby("symbol")["return"].transform(_roll_std)
    ret_long["mom_60d"] = ret_long.groupby("symbol")["return"].transform(_roll_mom)

    S = sentiment_long.copy()
    if not S.empty:
        S["date"] = pd.to_datetime(S["date"], errors="coerce")
        S = S.dropna(subset=["date"]).sort_values(["symbol","date"])
        if "sentiment_index" not in S.columns:
            alt = next((c for c in ["avg_sentiment","compound","vader_compound","sentiment"] if c in S.columns), None)
            S["sentiment_index"] = S[alt] if alt else 0.0
        S["rolling_sentiment_7d"] = S.groupby("symbol")["sentiment_index"] \
            .transform(lambda s: s.rolling(senti_window, min_periods=1).mean())
        if "news_count" in S.columns:
            S["news_count_7d"] = S.groupby("symbol")["news_count"] \
                .transform(lambda s: s.rolling(senti_window, min_periods=1).sum())
        else:
            S["news_count_7d"] = 0.0
        S_feat = S[["date","symbol","rolling_sentiment_7d","news_count_7d"]]
    else:
        S_feat = pd.DataFrame(columns=["date","symbol","rolling_sentiment_7d","news_count_7d"])

    feats = ret_long.merge(S_feat, on=["date","symbol"], how="left")
    return feats.sort_values(["symbol","date"])

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    DATA, RES = ROOT/"data", ROOT/"result"
    RES.mkdir(exist_ok=True)
    returns = pd.read_csv(DATA/"return_matrix.csv", parse_dates=["date"]).set_index("date").sort_index()
    senti = pd.read_csv(RES/"sentiment_index.csv") if (RES/"sentiment_index.csv").exists() else pd.DataFrame()
    feats = build_features(returns, senti)
    feats.to_csv(RES/"features_dataset.csv", index=False)
    print("💾 written:", (RES/"features_dataset.csv").resolve())
