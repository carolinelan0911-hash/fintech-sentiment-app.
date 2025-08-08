# run_all.py
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "generate_artifacts.py",          # base weights + features + static perf
    "make_sentiment_artifacts.py",    # weekly per-symbol + sentiment_index.csv
    "generate_sentiment_backtest.py", # baseline vs sentiment backtest
    "build_report_artifacts.py",      # correlations + heatmap + perf copies
]

def run(script: str) -> int:
    p = ROOT / script
    if not p.exists():
        print(f"[SKIP] {script} not found.")
        return 0
    print(f"[RUN ] {script}")
    return subprocess.call([sys.executable, str(p)], cwd=str(ROOT))

if __name__ == "__main__":
    rc = 0
    for s in SCRIPTS:
        rc |= run(s)
    print("\n✅ All steps attempted.")
    print("Next:  python -m streamlit run app.py --server.port 8501")
    sys.exit(rc)
