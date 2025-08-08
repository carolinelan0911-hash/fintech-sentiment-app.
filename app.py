# app.py
from __future__ import annotations

import os, sys, subprocess
from pathlib import Path
from typing import Iterable, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------- Page & Paths --------------------
st.set_page_config(
    page_title="Sentiment-Enhanced Crypto Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RES  = ROOT / "result"
NEWS_FIGS = RES / "news_figs"

PERF_CSV = RES / "portfolio_performance.csv"
MVO_W_CSV = RES / "mvo_weights.csv"
MVP_W_CSV = RES / "mvp_weights.csv"
RP_W_CSV  = RES / "risk_parity_weights.csv"
SENTI_CSV = RES / "sentiment_index.csv"

BACKTEST_CSVS = [
    RES / "backtest_results_sentiment.csv",
    RES / "backtest_baseline_vs_sentiment.csv",
    RES / "backtest_results.csv",
]
BACKTEST_IMGS = [RES / "fig_portfolio_vs_benchmark.png", RES / "fig_baseline_vs_sentiment.png"]

SCRIPTS = [
    ROOT / "generate_artifacts.py",
    ROOT / "make_sentiment_artifacts.py",
    ROOT / "generate_sentiment_backtest.py",
    ROOT / "build_report_artifacts.py",
]

# -------------------- Utils --------------------
def _first(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def safe_read_csv(path: Path, **kw) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", **kw)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def read_csv_with_date(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    if df.empty:
        return df
    lowers = {str(c).lower(): c for c in df.columns}
    date_col = next((lowers[c] for c in ("date", "datetime", "timestamp", "time") if c in lowers), None)
    if date_col is None:
        return df
    dt = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(df[date_col], errors="coerce")
    df["date"] = dt
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def run_script(script: Path) -> tuple[bool, str]:
    if not script.exists():
        return False, f"Script not found: {script.name}"
    try:
        subprocess.check_call([sys.executable, str(script)], cwd=str(ROOT))
        return True, f"Ran {script.name}"
    except subprocess.CalledProcessError as e:
        return False, f"{script.name} exited with {e.returncode}"
    except Exception as e:
        return False, f"Failed: {e}"

def _pick_file(*cands: str) -> Optional[str]:
    for p in cands:
        if os.path.exists(p):
            return p
    return None

# -------------------- Header --------------------
st.title("📈 Sentiment-Enhanced Crypto Portfolio Dashboard")
st.caption(f"Working dir: `{ROOT}`  |  Results dir: `{RES}`")

# -------------------- Sidebar: artifacts & generators --------------------
with st.sidebar:
    st.subheader("Artifacts status")
    for p in [PERF_CSV, MVO_W_CSV, MVP_W_CSV, RP_W_CSV, SENTI_CSV, *BACKTEST_CSVS, *BACKTEST_IMGS]:
        exists = p.exists()
        size_kb = (p.stat().st_size / 1024) if exists else 0
        st.write(f"{'✅' if exists else '❌'} {p.relative_to(ROOT)} {'' if not exists else f'({size_kb:.1f} KB)'}")

    st.divider()
    st.subheader("Generate / Refresh")
    if st.button("Run all generators", use_container_width=True, type="primary"):
        msgs = []
        for sp in SCRIPTS:
            ok, msg = run_script(sp)
            msgs.append(("✅" if ok else "❌") + " " + msg)
        st.session_state["_gen_msgs"] = msgs
        st.success("Finished. Scroll below / refresh sections.")
    if "_gen_msgs" in st.session_state:
        for m in st.session_state["_gen_msgs"]:
            st.write(m)

# -------------------- Backtest Chart --------------------
st.header("Baseline vs Sentiment-Adjusted (Cumulative)")
bt_csv = _first(BACKTEST_CSVS)
bt_img = _first(BACKTEST_IMGS)

col_chart, col_info = st.columns([3, 2], gap="large")
with col_chart:
    if bt_csv:
        df = read_csv_with_date(bt_csv)
        if not df.empty:
            lowers = {c.lower(): c for c in df.columns}
            def pick(words):
                for lc, orig in lowers.items():
                    if all(w in lc for w in words):
                        return orig
                return None
            base_col = pick(("baseline",)) or pick(("baseline", "cum"))
            senti_col = pick(("sentiment",)) or pick(("sentiment", "cum"))
            if base_col and senti_col and "date" in df.columns:
                plot = (
                    df[["date", base_col, senti_col]]
                    .dropna()
                    .rename(columns={base_col: "Baseline", senti_col: "Sentiment"})
                    .set_index("date")
                )
                st.line_chart(plot)
            else:
                st.info("Backtest CSV loaded but couldn't detect columns.")
    elif bt_img:
        st.image(str(bt_img), caption=bt_img.name)
    else:
        st.warning("No backtest artifacts found yet. Click **Run all generators**.")

with col_info:
    st.markdown(
        """
**Notes**
- Artifacts must be under `result/`.
- Click *Run all generators* to (re)build artifacts.
- Download raw CSVs below.
"""
    )

# -------------------- Downloads --------------------
st.subheader("Downloads")
for p in [
    RES / "backtest_baseline_vs_sentiment.csv",
    RES / "perf_summary_sentiment.csv",
    RES / "sentiment_index.csv",
    RES / "portfolio_performance.csv",
]:
    if p.exists():
        with open(p, "rb") as fh:
            st.download_button(f"Download {p.name}", fh, file_name=p.name)

# -------------------- Performance Table --------------------
st.header("Full Performance Data")
perf = safe_read_csv(PERF_CSV, index_col=0)
st.dataframe(perf.dropna(how="all") if not perf.empty else perf, use_container_width=True)

# -------------------- Latest Portfolio Weights --------------------
st.header("Latest Portfolio Weights")

def _last_weights(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Case 1: single "weight" column (static vector)
    if df.shape[1] == 1 and df.columns[0].strip().lower() == "weight":
        out = df.iloc[:, 0].astype(float).to_frame("Weight").dropna().sort_values("Weight", ascending=False)
        out.index.name = "Asset"
        return out
    # Case 2: time-indexed weights
    try:
        df.index = pd.to_datetime(df.index, errors="coerce")
    except Exception:
        pass
    last = df.sort_index().iloc[-1]
    out = pd.DataFrame({"Weight": pd.to_numeric(last, errors="coerce")}).dropna().sort_values("Weight", ascending=False)
    out.index.name = "Asset"
    return out

tabs = st.tabs(["MVO", "MVP", "Risk Parity"])
for t, pth in zip(tabs, [MVO_W_CSV, MVP_W_CSV, RP_W_CSV]):
    with t:
        dfw = safe_read_csv(pth, index_col=0)
        st.dataframe(_last_weights(dfw) if not dfw.empty else dfw, use_container_width=True)

# -------------------- Sentiment Index (per symbol) --------------------
st.subheader("Sentiment Index (per symbol)")
sent_path = _pick_file(str(SENTI_CSV), "data/sentiment_index.csv")
if sent_path is None:
    st.info("No sentiment_index.csv found under result/ or data/.")
else:
    si = pd.read_csv(sent_path, parse_dates=["date"]).dropna(subset=["date"])
    if not {"date", "symbol", "sentiment_index"}.issubset(si.columns):
        st.warning("sentiment_index.csv missing required columns: date, symbol, sentiment_index")
    else:
        si = si.sort_values("date")
        symbols_avail = sorted(si["symbol"].astype(str).unique().tolist())
        sel = st.selectbox("Symbol", symbols_avail, index=(symbols_avail.index("BTC") if "BTC" in symbols_avail else 0))
        sub = si[si["symbol"] == sel]
        st.line_chart(sub.set_index("date")["sentiment_index"])

# -------------------- Top Headlines (optional) --------------------
st.subheader("Top Headlines (last 7 days)")
raw_path = _pick_file("result/news_last7days.csv", "data/stage_1_news_raw.csv", "result/news_clean_data/clean_news_timeseries.csv")
if raw_path:
    try:
        df_news = pd.read_csv(raw_path)
        # Normalize datetime
        if "date" in df_news.columns:
            dt = pd.to_datetime(df_news["date"], errors="coerce")
        elif "published_on" in df_news.columns:
            dt = pd.to_datetime(df_news["published_on"], unit="s", errors="coerce")
        else:
            dt = pd.NaT
        df_news["__dt"] = dt
        df_news = df_news.dropna(subset=["__dt"]).sort_values("__dt", ascending=False)

        # Pick a headline/text column
        text_cols = [c for c in ["headline", "title", "summary", "body", "text", "content"] if c in df_news.columns]
        if text_cols:
            df_news["__head"] = df_news[text_cols].astype(str).agg(" ".join, axis=1)
            st.write(df_news[["__dt", "__head"]].head(10).rename(columns={"__dt": "time", "__head": "headline"}))
        else:
            st.caption("No text fields found in news file.")
    except Exception:
        st.caption("Failed to parse news file.")
else:
    st.caption("Provide article-level news at result/news_last7days.csv or data/stage_1_news_raw.csv to show headlines.")

# ==============================================================
#                       Gemini Integration
# ==============================================================

# Load .env then configure Gemini (safe for both local & Cloud)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

with st.sidebar:
    st.subheader("💬 Ask AI (Gemini)")

if not GEMINI_API_KEY:
    st.sidebar.error("❌ GEMINI_API_KEY not detected. Set it in .env (local) or Streamlit Secrets (Cloud).")
else:
    genai.configure(api_key=GEMINI_API_KEY)

    # Controls
    symbol_ai = st.sidebar.selectbox("Select cryptocurrency", ["BTC", "ETH", "SOL"], key="ai_symbol")
    weeks_back_ai = st.sidebar.slider("Lookback period (weeks)", 4, 12, 6, key="ai_weeks")

    if st.sidebar.button("Generate sentiment analysis", use_container_width=True):
        try:
            # Read inputs
            sentiment_df = pd.read_csv(SENTI_CSV, parse_dates=["date"]) if SENTI_CSV.exists() else pd.DataFrame()
            perf_df = pd.read_csv(PERF_CSV) if PERF_CSV.exists() else pd.DataFrame()

            cutoff_date = datetime.today() - timedelta(weeks=weeks_back_ai)
            recent_sentiment = pd.DataFrame()
            if not sentiment_df.empty:
                recent_sentiment = sentiment_df[
                    (sentiment_df["symbol"].astype(str) == symbol_ai) & (sentiment_df["date"] >= cutoff_date)
                ].copy()

            # Build sentiment summary text
            if not recent_sentiment.empty and {"date", "sentiment_index"}.issubset(recent_sentiment.columns):
                sentiment_summary = recent_sentiment[["date", "sentiment_index"]].to_string(index=False)
            else:
                sentiment_summary = "No recent sentiment data available."

            # Headlines (best-effort)
            headlines = []
            news_file = _pick_file("result/news_last7days.csv", "data/stage_1_news_raw.csv", "result/news_clean_data/clean_news_timeseries.csv")
            if news_file:
                try:
                    news_df = pd.read_csv(news_file)
                    if "symbol" in news_df.columns:
                        subn = news_df[news_df["symbol"].astype(str) == symbol_ai]
                    else:
                        subn = news_df.copy()
                    # Pick headline-like column
                    hcol = next((c for c in ["headline", "title", "summary", "text", "content", "body"] if c in subn.columns), None)
                    if hcol:
                        headlines = subn[hcol].astype(str).head(5).tolist()
                except Exception:
                    pass
            if not headlines:
                headlines = ["No news data available"]

            # Variation check
            if recent_sentiment.empty or recent_sentiment.get("sentiment_index", pd.Series(dtype=float)).nunique() <= 1:
                st.sidebar.warning("⚠️ Sentiment variation is too low — default to a Hold-leaning analysis.")
                sentiment_note = "Insufficient sentiment variation — default recommendation: HOLD until more data is available."
            else:
                sentiment_note = ""

            # Optimized prompt
            prompt = f"""
You are acting as a professional cryptocurrency portfolio strategist.
Using the data provided below, generate a concise and insightful investment analysis.

=== DATA INPUT ===
Asset: {symbol_ai}
Lookback period: {weeks_back_ai} weeks
Recent sentiment index (date, score):
{sentiment_summary}

Recent news headlines (last 7 days):
{chr(10).join(['- ' + h for h in headlines])}

Additional note: {sentiment_note}

=== ANALYSIS REQUIREMENTS ===
1. Provide a **Sentiment Overview**:
   - Identify whether sentiment is trending up, down, or stable.
   - Mention any noticeable changes over the lookback period.

2. **Impact on Risk & Portfolio Weights**:
   - Explain how sentiment trends could affect volatility and portfolio allocation.
   - Mention potential risk adjustments (e.g., reduce exposure if sentiment worsens).

3. **Key News Drivers**:
   - Summarize the market narrative from the headlines.
   - Identify if news sentiment aligns or conflicts with the sentiment index.

4. **Actionable Recommendation**:
   - Suggest whether to Increase, Maintain, or Reduce allocation.
   - Justify the recommendation in 2–3 sentences.

=== OUTPUT STYLE ===
- Use clear headings: Sentiment Overview / Risk Impact / News Drivers / Recommendation.
- Be concise and factual, avoid hype or overly casual language.
- Keep total output under 180 words.
- Do NOT give price targets or speculative predictions.
"""

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

            st.sidebar.subheader("Gemini Investment Analysis:")
            st.sidebar.write(response.text)

        except Exception as e:
            st.sidebar.error(f"Gemini call failed: {e}")
