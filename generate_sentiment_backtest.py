# generate_sentiment_backtest.py
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA, RES = ROOT/"data", ROOT/"result"
RES.mkdir(parents=True, exist_ok=True)

def pref_path(rel: str) -> Path:
    p1 = DATA/rel; p2 = Path("/mnt/data")/rel
    return p1 if p1.exists() else p2

FEATURES_CSV = pref_path("features_dataset.csv")
RETURNS_CSV  = pref_path("return_matrix.csv")
SENTI_CSV    = pref_path("sentiment_index.csv")

MVO_W = RES/"mvo_weights.csv"
OUT_W0   = RES/"weights_baseline_expanded.csv"
OUT_WS   = RES/"weights_sentiment_adjusted.csv"
OUT_RET  = RES/"backtest_baseline_vs_sentiment.csv"
OUT_PERF = RES/"perf_summary_sentiment.csv"
OUT_FIG  = RES/"fig_baseline_vs_sentiment.png"

def safe_read_csv(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()

def ensure_sentiment_file() -> None:
    if (RES/"sentiment_index.csv").exists() or SENTI_CSV.exists(): return
    gen = ROOT/"make_sentiment_artifacts.py"
    if not gen.exists(): raise FileNotFoundError("sentiment_index.csv missing and generator not found.")
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_senti", str(gen))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.main()

def load_returns_matrix() -> pd.DataFrame:
    if RETURNS_CSV.exists():
        R = safe_read_csv(RETURNS_CSV, parse_dates=["date"], index_col="date"); return R.sort_index()
    df = safe_read_csv(FEATURES_CSV)
    if df.empty: raise FileNotFoundError("returns not found.")
    for c in ("date","symbol","return"):
        if c not in df.columns: raise ValueError("features_dataset.csv must have date,symbol,return")
    df["date"]=pd.to_datetime(df["date"], errors="coerce")
    return df.pivot(index="date", columns="symbol", values="return").sort_index()

def pick_senti_csv() -> Path:
    c1 = RES/"sentiment_index.csv"; c2 = SENTI_CSV
    return c1 if c1.exists() else c2

def load_sentiment_matrix(symbols: list[str], feature_col="sentiment_index") -> pd.DataFrame:
    ensure_sentiment_file()
    df = safe_read_csv(pick_senti_csv())
    sym_col = "symbol" if "symbol" in df.columns else ("coin" if "coin" in df.columns else None)
    if sym_col is None or "date" not in df.columns or feature_col not in df.columns:
        raise ValueError("sentiment_index.csv requires date,symbol(or coin),feature")
    df["date"]=pd.to_datetime(df["date"], errors="coerce")
    df=df.dropna(subset=["date",sym_col])
    S = df.pivot(index="date", columns=sym_col, values=feature_col).sort_index()
    return S.reindex(columns=symbols).fillna(0.0)

def load_base_weights(symbols: list[str]) -> pd.DataFrame:
    W_raw = pd.read_csv(MVO_W, index_col=0)
    if W_raw.shape[1]==1 and W_raw.columns[0].strip().lower()=="weight":
        static = W_raw.iloc[:,0].astype(float).reindex(symbols).fillna(0.0)
        W = pd.DataFrame([static.values], columns=symbols); W.index=pd.to_datetime(["1970-01-01"])
        return W
    W = W_raw.copy()
    if set(symbols).issubset(W.columns):
        try:
            W.index = pd.to_datetime(W.index, errors="coerce"); W=W[~W.index.isna()].sort_index()
        except Exception:
            first = W.iloc[0].reindex(symbols).astype(float).fillna(0.0)
            W = pd.DataFrame([first.values], columns=symbols); W.index=pd.to_datetime(["1970-01-01"])
        return W.reindex(columns=symbols, fill_value=0.0)
    static = W.iloc[:,0].astype(float).reindex(symbols).fillna(0.0)
    W = pd.DataFrame([static.values], columns=symbols); W.index=pd.to_datetime(["1970-01-01"])
    return W

def align_weights_to_returns(W: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    W = W.copy()
    if not isinstance(W.index, pd.DatetimeIndex): W.index = pd.to_datetime(["1970-01-01"])
    W = W.reindex(index=R.index).sort_index().ffill().bfill().clip(lower=0.0)
    row_sum = W.sum(axis=1).replace(0.0, np.nan)
    W = W.div(row_sum, axis=0).fillna(0.0)
    return W.reindex(columns=R.columns, fill_value=0.0)

def adjust_weights_with_sentiment(W: pd.DataFrame, S: pd.DataFrame, scale=0.25, wmin=0.0, wmax=0.30) -> pd.DataFrame:
    idx = W.index.intersection(S.index)
    W = W.loc[idx]; X = S.loc[idx].fillna(0.0)
    W_adj = W * (1.0 + scale*X); W_adj = W_adj.clip(lower=wmin, upper=wmax)
    row_sum = W_adj.sum(axis=1).replace(0.0, np.nan)
    return W_adj.div(row_sum, axis=0).fillna(0.0).reindex(columns=W.columns, fill_value=0.0)

def returns_from_weights(W: pd.DataFrame, R: pd.DataFrame, txn_cost_bps=10.0) -> pd.Series:
    tc = txn_cost_bps/10000.0
    idx = R.index.intersection(W.index); W=W.loc[idx]; R=R.loc[idx]
    prev=None; out=[]
    for dt in idx:
        w=W.loc[dt].values; r=R.loc[dt].values
        turnover = 0.0 if prev is None else float(np.sum(np.abs(w-prev)))
        ret = float(np.nansum(w*r)) - tc*turnover
        out.append(ret); prev=w
    return pd.Series(out, index=idx)

def main(scale=0.25, txn_cost_bps=10.0, sentiment_feature="sentiment_index"):
    # returns + symbols
    R = load_returns_matrix(); symbols = R.columns.tolist()
    # baseline
    W0_raw = load_base_weights(symbols); W0 = align_weights_to_returns(W0_raw, R); W0.to_csv(OUT_W0)
    # sentiment
    S = load_sentiment_matrix(symbols, feature_col=sentiment_feature).reindex(index=R.index).fillna(0.0)
    # adjusted weights
    WS = adjust_weights_with_sentiment(W0, S, scale=scale, wmin=0.0, wmax=0.30); WS.to_csv(OUT_WS)
    # backtest
    r_base = returns_from_weights(W0, R, txn_cost_bps=txn_cost_bps).rename("Baseline")
    r_sent = returns_from_weights(WS, R, txn_cost_bps=txn_cost_bps).rename("Sentiment")
    cmp = pd.concat([r_base, r_sent], axis=1); cmp.to_csv(OUT_RET)
    # performance
    eq_base = (1+r_base.dropna()).cumprod(); eq_sent=(1+r_sent.dropna()).cumprod()
    perf = pd.DataFrame({
        "Strategy":["Baseline","Sentiment"],
        "Total Return":[float(eq_base.iloc[-1]-1.0) if not eq_base.empty else np.nan,
                        float(eq_sent.iloc[-1]-1.0) if not eq_sent.empty else np.nan],
        "Sharpe":[float(r_base.mean()/r_base.std(ddof=1)*np.sqrt(52)) if r_base.std(ddof=1)>0 else np.nan,
                  float(r_sent.mean()/r_sent.std(ddof=1)*np.sqrt(52)) if r_sent.std(ddof=1)>0 else np.nan],
        "MaxDrawdown":[float(((eq_base/eq_base.cummax())-1).min()) if not eq_base.empty else np.nan,
                       float(((eq_sent/eq_sent.cummax())-1).min()) if not eq_sent.empty else np.nan],
        "Vol":[float(r_base.std(ddof=1)) if r_base.notna().any() else np.nan,
               float(r_sent.std(ddof=1)) if r_sent.notna().any() else np.nan]
    })
    perf.to_csv(OUT_PERF, index=False)
    # figure
    fig, ax = plt.subplots(figsize=(8,4))
    if not eq_base.empty: ax.plot(eq_base.index, eq_base.values, label="Baseline")
    if not eq_sent.empty: ax.plot(eq_sent.index, eq_sent.values, label="Sentiment")
    ax.set_title("Cumulative Return — Baseline vs Sentiment"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=150); plt.close(fig)
    print("💾 wrote:", OUT_W0, OUT_WS, OUT_RET, OUT_PERF, OUT_FIG, sep="\n")

if __name__ == "__main__":
    main()
