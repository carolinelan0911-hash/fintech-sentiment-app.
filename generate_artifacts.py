# generate_artifacts.py
from pathlib import Path
import os, pandas as pd, numpy as np
from portfolio_model import (
    calculate_mvo_weights, calculate_min_var_weights, calculate_risk_parity_weights,
    backtest_portfolio, evaluate_performance
)
from features_dataset import build_features

ROOT = Path(__file__).resolve().parent
DATA, RES = ROOT/"data", ROOT/"result"
RES.mkdir(exist_ok=True)

def pick_path(*cands):
    for p in cands:
        p = Path(p)
        if p.exists(): return p
    raise FileNotFoundError("None of the candidate paths exist.")

returns_path = pick_path(RES/"return_matrix.csv", DATA/"return_matrix.csv")
returns = pd.read_csv(returns_path, parse_dates=["date"]).set_index("date").sort_index()
num_cols = returns.select_dtypes(include=[np.number]).columns.tolist()
returns = returns[num_cols].dropna(how="all")
print(f"✅ Returns matrix shape: {returns.shape}")

senti_path = RES/"sentiment_index.csv"
sent_long = pd.read_csv(senti_path) if senti_path.exists() else pd.DataFrame()

# features
features_out = RES/"features_dataset.csv"
if (not features_out.exists()) or (os.getenv("REBUILD_FEATURES","0")=="1"):
    feats = build_features(returns, sent_long)
    feats.to_csv(features_out, index=False); print("💾 features_dataset.csv written:", features_out.resolve())

# inputs to optimizers
C = returns.cov().fillna(0.0); mu = returns.mean().fillna(0.0)

# weights with fallbacks
try:
    mvo_w = calculate_mvo_weights(C, mu)
except Exception:
    try: mvo_w = calculate_min_var_weights(C)
    except Exception: mvo_w = pd.Series(np.ones(len(C))/len(C), index=C.columns)

try:
    mvp_w = calculate_min_var_weights(C)
except Exception:
    mvp_w = pd.Series(np.ones(len(C))/len(C), index=C.columns)

try:
    rp_w = calculate_risk_parity_weights(C)
except Exception as e:
    print("⚠️ Risk parity failed/disabled:", e); rp_w=None

mvo_w.rename("Weight").to_csv(RES/"mvo_weights.csv")
mvp_w.rename("Weight").to_csv(RES/"mvp_weights.csv")
if rp_w is not None: rp_w.rename("Weight").to_csv(RES/"risk_parity_weights.csv")
print("💾 weights written.")

# static backtests (no TC since static weights don't turnover)
mvo_ret = backtest_portfolio(mvo_w, returns, txn_cost_bps=0.0)
mvp_ret = backtest_portfolio(mvp_w, returns, txn_cost_bps=0.0)
pd.DataFrame(mvo_ret).to_csv(RES/"mvo_returns.csv")
pd.DataFrame(mvp_ret).to_csv(RES/"mvp_returns.csv")
if rp_w is not None:
    rp_ret = backtest_portfolio(rp_w, returns, txn_cost_bps=0.0)
    pd.DataFrame(rp_ret).to_csv(RES/"risk_parity_returns.csv")

perf = pd.DataFrame({
    "MVO": evaluate_performance(mvo_ret),
    "MVP": evaluate_performance(mvp_ret),
    **({"Risk Parity": evaluate_performance(rp_ret)} if rp_w is not None else {})
}).T
perf.to_csv(RES/"portfolio_performance.csv")
print("💾 portfolio_performance.csv")
