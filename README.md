# fintech-sentiment-app.
Crypto Sentiment Portfolio Optimizer (with Google Gemini AI)
This project builds an end-to-end cryptocurrency portfolio optimization system using historical performance, sentiment analysis, and AI-powered insights from Google Gemini.
It includes scripts for data preparation, sentiment backtesting, portfolio analysis, and an interactive Streamlit web app.

🚀 Features
Portfolio optimization models — MVO, MVP, Risk Parity.
Sentiment integration — Adjusts weights based on sentiment index per asset.
Backtesting — Compare baseline vs sentiment-adjusted portfolios.
Gemini AI insights — Natural language investment analysis combining sentiment data and news.
Visualization — Heatmaps, performance charts, and benchmark comparisons.
One-click pipeline — run_all.py executes the full artifact generation workflow.

📂 Project Structure
bash
.
├── app.py                      # Streamlit app (frontend)
├── portfolio_model.py          # Portfolio optimization models
├── sentiment_portfolio.py      # Sentiment-based weight adjustments & plots
├── sentiment_pipeline.py       # Data pipeline for sentiment computation
├── features_dataset.py         # Feature engineering
├── generate_artifacts.py       # Generates base weights & static performance
├── make_sentiment_artifacts.py # Generates weekly sentiment index per symbol
├── generate_sentiment_backtest.py # Runs sentiment vs baseline backtest
├── build_report_artifacts.py   # Creates correlations, heatmaps, and copies results
├── run_all.py                  # Executes all scripts in sequence
├── result/                     # Generated CSVs, PNGs, reports
└── data/                       # Raw input datasets


## 🛠 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/carolinelan0911-hash/fintech-sentiment-app.git
cd fintech-sentiment-app
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run locally
streamlit run app.py
🌐 Deployment
This app is deployed using Streamlit Cloud.
