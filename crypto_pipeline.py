
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional
import random

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

API_KEY = "92168f0a2d117950d5a46c5d2ef0e3b0fc724a81e27f70ddecec02f9e00376e6"
TOP_COINS = ["BTC", "ETH", "XRP", "BNB", "SOL", "DOGE", "ADA", "DOT", "TRX", "AVAX"]

def _ensure_dir(root: Path, sub: str | Path) -> Path:
    path = root / sub
    path.mkdir(parents=True, exist_ok=True)
    return path

def build_week_dirs(
    week_folder: str | Path = "week_crypto",
    results_folder: str = "results",
    data_sub: str = "clean_data",
) -> Path:
    project_root = Path.cwd().resolve()
    week_root = project_root if project_root.name == str(week_folder) else _ensure_dir(project_root, week_folder)
    results_root = _ensure_dir(week_root, results_folder)
    data_dir = _ensure_dir(results_root, data_sub)
    return data_dir

def get_daily_ohlcv(
    symbol: str,
    api_key: str,
    limit: int = 2000,
    currency: str = "USD",
) -> Optional[pd.DataFrame]:
    url = (f"https://min-api.cryptocompare.com/data/v2/histoday"
           f"?fsym={symbol}&tsym={currency}&limit={limit}&api_key={api_key}")
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if "Data" not in data or "Data" not in data["Data"]:
            logging.warning("No data for %s: %s", symbol, data)
            return None

        df = pd.DataFrame(data["Data"]["Data"])
        if df.empty:
            logging.warning("Empty data for %s", symbol)
            return None

        df["date"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volumeto": "usd_volume",
            "volumefrom": "btc_volume"
        })
        df["symbol"] = symbol
        df = df[["date", "open", "high", "low", "close", "usd_volume", "btc_volume", "symbol"]]
        df.set_index(["symbol", "date"], inplace=True)
        return df
    except Exception as e:
        logging.error("Error fetching data for %s: %s", symbol, e)
        return None

def stage1_etl(
    api_key: str,
    symbols: List[str],
    history_limit: int = 2000,
    currency: str = "USD",
    sleep_sec: float = 1,
    data_dir: Path | None = None,
    filename: str = "stage_1_crypto_data.csv",
) -> pd.DataFrame:
    logging.info("Starting Stage 1 ETL ...")
    all_frames = []
    for i, sym in enumerate(symbols, 1):
        logging.info("Downloading history for %s (%d/%d)", sym, i, len(symbols))
        df = get_daily_ohlcv(sym, api_key, history_limit, currency)
        if df is not None:
            all_frames.append(df)
        time.sleep(sleep_sec + random.uniform(0.0, 0.5))
    if not all_frames:
        raise RuntimeError("No historical data retrieved.")
    data = pd.concat(all_frames).sort_index()
    if data_dir is not None:
        out_path = data_dir / filename
        data.to_csv(out_path)
        logging.info("Stage 1 CSV written to %s", out_path)
    return data

def stage2_feature_engineering(
    tidy_prices: pd.DataFrame | None = None,
    csv_path: Path | None = None,
    data_dir: Path | None = None,
    filename: str = "stage_2_crypto_data.csv",
) -> pd.DataFrame:
    if tidy_prices is None:
        if csv_path is None:
            raise ValueError("Provide either tidy_prices or csv_path.")
        logging.info("Reading Stage 1 CSV from %s", csv_path)
        tidy_prices = pd.read_csv(csv_path, index_col=["symbol", "date"], parse_dates=["date"])

    df = tidy_prices.reset_index().sort_values(["symbol", "date"]).copy()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
    df['usd_volume'] = pd.to_numeric(df['usd_volume'], errors='coerce')
    df.loc[df['usd_volume'] <= 0, 'usd_volume'] = np.nan
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    for m in [7, 14, 21, 28, 42]:
        rolling_mean = df.groupby("symbol")["usd_volume"].shift(1).rolling(window=m, min_periods=m).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"v_{m}d"] = np.log(df["usd_volume"]) - np.log(rolling_mean)
        df[f"v_{m}d"] = df[f"v_{m}d"].replace([np.inf, -np.inf], np.nan)

    df["log_return"] = np.log1p(df.groupby("symbol")["close"].pct_change())
    df = df.replace([-np.inf, np.inf], np.nan)

    for m in [14, 21, 28, 42, 90]:
        shifted = df.groupby("symbol")["log_return"].shift(7)
        df[f"momentum_{m}"] = np.exp(shifted.rolling(m, min_periods=m).sum()) - 1.0
        df[f"volatility_{m}"] = (
            df.groupby("symbol")["log_return"]
            .rolling(m, min_periods=m)
            .std()
            .reset_index(level=0, drop=True)
        ) * np.sqrt(365.0)

    df["strev_daily"] = df["log_return"]

    dfw = (
        df.set_index("date")
        .groupby("symbol", group_keys=False)
        .resample("W-WED")
        .last()
    )
    dfw["return"] = dfw.groupby("symbol")["close"].pct_change()
    dfw["strev_weekly"] = dfw["return"]
    dfw = dfw.reset_index()
    dfw = dfw.replace([np.inf, -np.inf], np.nan)

    if data_dir is not None:
        out_path = data_dir / filename
        dfw.to_csv(out_path, index=False)
        logging.info("Stage 2 CSV written to %s", out_path)
        stats = dfw.describe().T
        stats.to_csv(data_dir / "descriptive_stats.csv")
        return_matrix = dfw.pivot(index="date", columns="symbol", values="return")
        return_matrix.to_csv(data_dir / "return_matrix.csv")
    return dfw

def run_pipeline():
    data_dir = build_week_dirs("week_crypto")
    df_prices = stage1_etl(API_KEY, TOP_COINS, history_limit=2000, currency="USD", data_dir=data_dir)
    stage2_feature_engineering(df_prices, data_dir=data_dir)
    print("Done! Data saved in:", data_dir.resolve())

if __name__ == "__main__":
    run_pipeline()
