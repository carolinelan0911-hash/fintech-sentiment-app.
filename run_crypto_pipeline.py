from crypto_pipeline import (
    stage1_etl,
    stage2_feature_engineering,
    build_week_dirs
)

API_KEY = "92168fa02d117950d5a4e60d3eb0fc724a81e27f70ddecce02f9e00376e6"
PAGES = [1, 2]
TOP_LIMIT = 100
HISTORY_LIMIT = 2000
CURRENCY = "USD"
WEEK_FOLDER = "week5_crypto"

def main() -> None:
    data_dir = build_week_dirs(WEEK_FOLDER)

    df_prices = stage1_etl(
        api_key=API_KEY,
        symbols=None,
        history_limit=HISTORY_LIMIT,
        currency=CURRENCY,
        pages=PAGES,
        top_limit=TOP_LIMIT,
        data_dir=data_dir
    )

    df_features = stage2_feature_engineering(
        prices_df=df_prices,
        output_dir=data_dir
    )

    print("Pipeline completed.")

if __name__ == "__main__":
    main()
