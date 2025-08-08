# sentiment_pipeline.py
from __future__ import annotations
import pandas as pd

__all__ = ["calculate_sentiment_index"]

def _normalize_symbol_col(df: pd.DataFrame) -> pd.DataFrame:
    for cand in ("symbol","coin","asset","ticker"):
        if cand in df.columns:
            if cand != "symbol": df = df.rename(columns={cand:"symbol"})
            break
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper().str.replace(r"[^A-Z0-9]","", regex=True)
    return df

def _pick_sent_col(cols: list[str]) -> str|None:
    for c in ["sentiment_index","avg_sentiment","compound","vader_compound","sentiment"]:
        if c in cols: return c
    return None

def _ewma(x: pd.Series, decay: float) -> pd.Series:
    alpha = 1.0 - float(decay)
    alpha = min(max(alpha, 1e-6), 0.999999)
    return pd.to_numeric(x, errors="coerce").ewm(alpha=alpha, adjust=False).mean()

def calculate_sentiment_index(df_weekly: pd.DataFrame,
                              decay_factor: float = 0.8,
                              target_symbols: list[str]|None = None) -> pd.DataFrame:
    if "date" not in df_weekly.columns:
        raise ValueError("Input must contain 'date'.")
    df = df_weekly.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = _normalize_symbol_col(df)

    if "symbol" in df.columns:
        sent_col = _pick_sent_col(df.columns.tolist())
        if sent_col is None: raise ValueError("No sentiment column found.")
        cols = ["date","symbol",sent_col] + (["news_count"] if "news_count" in df.columns else [])
        df = df[cols].rename(columns={sent_col:"sent_raw"}).copy()
        outs=[]
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_values("date")
            ema = _ewma(g["sent_raw"], decay_factor)
            out = pd.DataFrame({
                "date":g["date"], "symbol":sym,
                "sentiment_index":ema.values,
                "sentiment_ema_7d":ema.values,
                "sentiment_change":ema.diff().fillna(0.0).values,
            })
            if "news_count" in g.columns:
                out["news_count"] = pd.to_numeric(g["news_count"], errors="coerce").fillna(0.0).values
            outs.append(out)
        res = pd.concat(outs, ignore_index=True)
        return res.sort_values(["symbol","date"]).reset_index(drop=True)

    value_cols = [c for c in df.columns if c.lower() not in ("date","news_count")]
    if not value_cols: raise ValueError("No usable sentiment columns.")
    if len(value_cols)==1:
        if not target_symbols: raise ValueError("Market-only input; provide target_symbols.")
        sent_col = value_cols[0]
        market = df[["date", sent_col] + (["news_count"] if "news_count" in df.columns else [])] \
                   .rename(columns={sent_col:"avg_sentiment"})
        keys = pd.DataFrame({"symbol":[s.upper() for s in target_symbols]})
        market["key"]=1; keys["key"]=1
        long_df = market.merge(keys, on="key").drop(columns="key")
        return calculate_sentiment_index(long_df, decay_factor=decay_factor)

    long_df = df.melt(id_vars=["date"], value_vars=value_cols,
                      var_name="symbol", value_name="avg_sentiment")
    if "news_count" in df.columns:
        long_df = long_df.merge(df[["date","news_count"]], on="date", how="left")
    return calculate_sentiment_index(long_df, decay_factor=decay_factor)
