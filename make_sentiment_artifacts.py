# make_sentiment_artifacts.py
"""
Build per-symbol weekly sentiment and a smoothed sentiment index.

Inputs supported:
1) Article-level with text -> compute VADER compound, tag symbols, weekly-aggregate
2) Long weekly table (date, symbol, avg_sentiment[, news_count]) -> pass-through
3) Wide weekly table (date + one column per symbol) -> melt to long

Outputs:
- result/news_clean_data/clean_news_timeseries.csv
- result/sentiment_index.csv
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict
import os, re, sys, subprocess
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT  = ROOT / "result"
(OUT / "news_clean_data").mkdir(parents=True, exist_ok=True)

RETURNS_CSV = DATA / "return_matrix.csv"
NEWS_SOURCES = [
    DATA / "stage_1_news_raw.csv",
    DATA / "clean_news_timeseries.csv",
    OUT  / "news_clean_data" / "clean_news_timeseries.csv",
]

MIN_NEWS = 3   # sparse-news gate

def _pick_existing(cands: Iterable[Path]) -> Path:
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("No news file found in: " + " | ".join(map(str, cands)))

def _load_symbols() -> List[str]:
    majors = ["BTC","ETH","ADA","XRP","BNB","TRX","SOL","DOT","DOGE","AVAX"]
    if RETURNS_CSV.exists():
        df = pd.read_csv(RETURNS_CSV, parse_dates=["date"])
        cols = [c for c in df.columns if c.lower() != "date"]
        return cols if cols else majors
    return majors

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("date", "published_on", "datetime", "timestamp", "time"):
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            return df.assign(date=dt).dropna(subset=["date"]).sort_values("date")
    raise ValueError("No date-like column in news file.")

def _auto_install_vader() -> None:
    try:
        import vaderSentiment  # noqa
    except Exception:
        print("[INFO] Installing vaderSentiment ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment"], stdout=sys.stdout)

def _maybe_compute_compound(df: pd.DataFrame) -> pd.DataFrame:
    need = ("compound" not in df.columns)
    if not need:
        try:
            uniq = pd.to_numeric(df["compound"], errors="coerce").nunique(dropna=True)
            need = (uniq <= 1)
        except Exception:
            need = True
    text_col = next((c for c in ("body","content","text","summary","title") if c in df.columns), None)
    if not need or text_col is None:
        return df
    _auto_install_vader()
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception as e:
        print("[WARN] VADER import failed, skip compound scoring:", e); return df
    sid = SentimentIntensityAnalyzer()
    join_cols = [c for c in ("title","summary","body","content","text") if c in df.columns]
    txt = df[join_cols].astype(str).agg(" ".join, axis=1)
    df = df.copy()
    df["compound"] = txt.map(lambda s: float(sid.polarity_scores(s)["compound"]))
    return df

def _normalize_symbol_col(df: pd.DataFrame) -> pd.DataFrame:
    if "symbol" in df.columns:
        s = df["symbol"].astype(str).str.upper()
        return df.assign(symbol=s)
    if "coin" in df.columns:
        s = df["coin"].astype(str).str.upper()
        return df.assign(symbol=s).drop(columns=["coin"])
    return df

def _alias_map(symbols: List[str]) -> Dict[str, List[str]]:
    base = {
        "BTC":["BTC","BITCOIN"], "ETH":["ETH","ETHEREUM"],
        "ADA":["ADA","CARDANO"], "XRP":["XRP","RIPPLE"], "BNB":["BNB","BINANCE COIN","BINANCECOIN"],
        "DOGE":["DOGE","DOGECOIN"], "MATIC":["MATIC","POLYGON"], "DOT":["DOT","POLKADOT"],
        "AVAX":["AVAX","AVALANCHE"], "ATOM":["ATOM","COSMOS"], "LTC":["LTC","LITECOIN"],
        "LINK":["LINK","CHAINLINK"], "UNI":["UNI","UNISWAP"], "AAVE":["AAVE"], "SUI":["SUI"],
        "TRX":["TRX","TRON"], "ETC":["ETC","ETHEREUM CLASSIC","ETHEREUMCLASSIC"],
        "BCH":["BCH","BITCOIN CASH","BITCOINCASH"], "FIL":["FIL","FILECOIN"],
        "OP":["OP","OPTIMISM"], "ARB":["ARB","ARBITRUM"], "NEAR":["NEAR"], "TON":["TON","TONCOIN"],
    }
    out: Dict[str, List[str]] = {}
    for s in symbols:
        u = s.upper()
        out[u] = sorted(set([u, f"${u}", f"#{u}"] + base.get(u, [])))
    return out

def _tag_to_symbols(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    df = _normalize_symbol_col(df)
    if "symbol" in df.columns:
        return df
    text_cols = [c for c in ("title","body","content","summary") if c in df.columns]
    if not text_cols:
        return pd.DataFrame(columns=list(df.columns)+["symbol"])
    txt = df[text_cols].astype(str).agg(" ".join, axis=1).str.upper()
    alias = _alias_map(symbols)
    patterns = {sym: re.compile(r"(?:^|\s)(" + r"|".join(map(re.escape, incs)) + r")(?:\s|$)") for sym, incs in alias.items()}
    syms = []
    for s in txt:
        hit = None
        for sym, rgx in patterns.items():
            if rgx.search(s):
                hit = sym; break
        syms.append(hit)
    out = df.copy(); out["symbol"] = syms
    out = out.dropna(subset=["symbol"]); out["symbol"] = out["symbol"].astype(str).str.upper()
    return out

def main() -> None:
    symbols = _load_symbols()
    news_path = _pick_existing(NEWS_SOURCES)
    print("Using news source:", news_path)

    df_raw = pd.read_csv(news_path)
    df_raw = _ensure_datetime(df_raw)

    lower = {c.lower(): c for c in df_raw.columns}

    # Case 1: long weekly with symbol + avg_sentiment
    if ("symbol" in lower or "coin" in lower) and ("avg_sentiment" in lower or "sentiment_index" in lower):
        df = _normalize_symbol_col(df_raw.copy())
        sent_col = "avg_sentiment" if "avg_sentiment" in df.columns else "sentiment_index"
        keep = ["date","symbol",sent_col] + (["news_count"] if "news_count" in df.columns else [])
        weekly_symbol = df[keep].rename(columns={sent_col:"avg_sentiment"}).copy()
        if "news_count" not in weekly_symbol.columns:
            weekly_symbol["news_count"] = 1
        weekly_symbol = weekly_symbol.sort_values(["symbol","date"])

    else:
        # Case 2: wide (date + one column per asset)
        overlap_assets = [c for c in df_raw.columns if c.upper() in {s.upper() for s in symbols}]
        if "date" in df_raw.columns and overlap_assets and ("symbol" not in df_raw.columns):
            wide = df_raw[["date"] + overlap_assets].copy()
            weekly_symbol = wide.melt("date", var_name="symbol", value_name="avg_sentiment")
            weekly_symbol["symbol"] = weekly_symbol["symbol"].astype(str).str.upper()
            weekly_symbol["news_count"] = 1
            weekly_symbol = weekly_symbol.dropna(subset=["avg_sentiment"]).sort_values(["symbol","date"])
        else:
            # Case 3: article-level
            df = _maybe_compute_compound(df_raw)
            tagged = _tag_to_symbols(df, symbols)
            if tagged.empty:
                weekly_symbol = pd.DataFrame(columns=["date","symbol","avg_sentiment","news_count"])
            else:
                tagged["week"] = tagged["date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
                weekly_symbol = (
                    tagged.groupby(["week","symbol"], as_index=False)
                          .agg(avg_sentiment=("compound","mean"), news_count=("compound","size"))
                          .rename(columns={"week":"date"})
                          .sort_values(["symbol","date"])
                )

    # sparse news gate
    weekly_symbol["news_count"] = pd.to_numeric(weekly_symbol.get("news_count", 1), errors="coerce").fillna(0).astype(int)
    weekly_symbol.loc[weekly_symbol["news_count"] < MIN_NEWS, "avg_sentiment"] = 0.0

    weekly_out = OUT / "news_clean_data" / "clean_news_timeseries.csv"
    weekly_symbol.to_csv(weekly_out, index=False, encoding="utf-8")
    print("💾 wrote weekly per-symbol:", weekly_out.resolve())

    # Build smoothed index
    from sentiment_pipeline import calculate_sentiment_index
    senti_long = calculate_sentiment_index(weekly_symbol.rename(columns={"avg_sentiment":"sentiment"}), decay_factor=0.8)
    senti_long.to_csv(OUT / "sentiment_index.csv", index=False, encoding="utf-8")
    print("💾 wrote sentiment index:", (OUT / "sentiment_index.csv").resolve())

if __name__ == "__main__":
    main()
