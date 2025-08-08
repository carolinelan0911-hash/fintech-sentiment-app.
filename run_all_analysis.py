from crypto_pipeline import (
    stage1_etl,
    stage2_feature_engineering,
    build_week_dirs,
)

API_KEY = "92168f0a2d117950d5a46c5d2ef0e3b0fc724a81e27f70ddecec02f9e00376e6"
TOP_COINS = ["BTC", "ETH", "XRP", "BNB", "SOL", "DOGE", "ADA", "DOT", "TRX", "AVAX"]
HISTORY_LIMIT = 2000
CURRENCY = "USD"
WEEK_FOLDER = "week5_crypto"


def main() -> None:
    data_dir = build_week_dirs(WEEK_FOLDER)

    # Stage 1 – fetch daily OHLCV for fixed TOP_COINS
    df_prices = stage1_etl(
        api_key=API_KEY,
        symbols=TOP_COINS,  # <--- Use this instead of 'pages'
        history_limit=HISTORY_LIMIT,
        currency=CURRENCY,
        data_dir=data_dir,
    )

    # Stage 2 – compute features
    stage2_feature_engineering(
        tidy_prices=df_prices,
        data_dir=data_dir,
    )

    print("Done! Data saved in:", data_dir.resolve())


if __name__ == "__main__":
    main()
