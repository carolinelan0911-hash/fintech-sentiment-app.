# portfolio_model.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _safe_cov(X: np.ndarray) -> np.ndarray:
    if X.size == 0: return np.zeros((0,0))
    X = np.asarray(X, float)
    X[~np.isfinite(X)] = np.nan
    mu = np.nanmean(X, axis=0)
    Xc = X - mu
    Xc[~np.isfinite(Xc)] = 0.0
    T = X.shape[0]
    C = (Xc.T @ Xc) / max(T - 1, 1)
    diag = np.diag(C)
    if (not np.all(np.isfinite(diag))) or (np.min(np.abs(diag)) < 1e-12):
        C = C + np.eye(C.shape[0]) * 1e-6
    return C

def _clip_norm(w: pd.Series | np.ndarray, wmin=0.0, wmax=1.0) -> pd.Series:
    if isinstance(w, pd.Series):
        idx = w.index; v = w.to_numpy(float)
    else:
        idx = None; v = np.asarray(w, float)
    v = np.clip(v, wmin, wmax)
    s = v.sum()
    v = (np.ones_like(v)/len(v)) if (s<=0 or not np.isfinite(s)) else v/s
    return pd.Series(v, index=idx)

def calculate_mvo_weights(C: pd.DataFrame|np.ndarray, mu: pd.Series|np.ndarray,
                          gamma=3.0, wmin=0.0, wmax=0.30) -> pd.Series:
    if isinstance(C, pd.DataFrame):
        idx = list(C.columns); C = C.to_numpy(float)
    else:
        C = np.asarray(C, float); idx = list(range(C.shape[0]))
    mu = np.asarray(mu, float).reshape(-1)
    C = C + np.eye(C.shape[0])*1e-8
    try: invC = np.linalg.inv(C)
    except np.linalg.LinAlgError: invC = np.linalg.pinv(C)
    raw = invC @ mu
    if not np.isfinite(raw).any() or np.allclose(raw,0): raw = np.ones_like(raw)
    return _clip_norm(pd.Series(raw/gamma, index=idx), wmin, wmax)

def calculate_min_var_weights(C: pd.DataFrame|np.ndarray,
                              wmin=0.0, wmax=0.30) -> pd.Series:
    if isinstance(C, pd.DataFrame):
        idx = list(C.columns); C = C.to_numpy(float)
    else:
        C = np.asarray(C, float); idx = list(range(C.shape[0]))
    C = C + np.eye(C.shape[0])*1e-8
    try: invC = np.linalg.inv(C)
    except np.linalg.LinAlgError: invC = np.linalg.pinv(C)
    ones = np.ones(C.shape[0])
    return _clip_norm(pd.Series(invC @ ones, index=idx), wmin, wmax)

def calculate_risk_parity_weights(C: pd.DataFrame|np.ndarray,
                                  wmin=0.0, wmax=0.30,
                                  iters=500, tol=1e-8) -> pd.Series:
    if isinstance(C, pd.DataFrame):
        idx = list(C.columns); C = C.to_numpy(float)
    else:
        C = np.asarray(C, float); idx = list(range(C.shape[0]))
    n = C.shape[0]; w = np.ones(n)/n; C = C + np.eye(n)*1e-8
    for _ in range(iters):
        mrc = C @ w
        if not np.isfinite(mrc).any(): break
        target = (w*mrc).mean()
        scale = np.where(mrc>0, target/(mrc+1e-12), 1.0)
        w = w * scale
        w = np.clip(w, wmin, wmax); w = w / w.sum()
        if np.max(np.abs(w*(C@w) - target)) < tol: break
    return pd.Series(w, index=idx)

def backtest_portfolio(weights: pd.Series|pd.DataFrame,
                       returns_df: pd.DataFrame,
                       txn_cost_bps: float = 0.0) -> pd.Series:
    R = returns_df.copy()
    tc = txn_cost_bps/10000.0
    if isinstance(weights, pd.Series):
        w = weights.reindex(R.columns).fillna(0.0).to_numpy(float)
        port = (R.to_numpy(float) * w).sum(axis=1)
        return pd.Series(port, index=R.index, name="Portfolio")
    W = weights.reindex(index=R.index, columns=R.columns).fillna(0.0)
    prev = None; out=[]
    for dt, row in R.iterrows():
        w = W.loc[dt].to_numpy(float); r = row.to_numpy(float)
        turnover = 0.0 if prev is None else float(np.sum(np.abs(w-prev)))
        out.append(float(np.nansum(w*r)) - tc*turnover)
        prev = w
    return pd.Series(out, index=R.index, name="Portfolio")

def evaluate_performance(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return pd.Series({"Mean": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDrawdown": np.nan, "CAGR": np.nan})

    mean = r.mean()
    vol = r.std(ddof=1)
    sharpe = (mean / vol * np.sqrt(52)) if vol and np.isfinite(vol) else np.nan

    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    mdd = ((eq / peak) - 1.0).min()

    cagr = eq.iloc[-1] ** (52.0 / len(eq)) - 1.0 if len(eq) > 0 and eq.iloc[0] > 0 else np.nan

    return pd.Series({"Mean": mean, "Vol": vol, "Sharpe": sharpe, "MaxDrawdown": mdd, "CAGR": cagr})
