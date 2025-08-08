"""
validate_dataset.py
Check data quality of final_dataset.csv:
- Missing values
- Date continuity
- Descriptive statistics
- NaN ratio and abnormal values
Generates dataset_validation.txt for summary.
"""

import pandas as pd
from pathlib import Path

# ------------------------- CONFIG -------------------------
FINAL_DATA_PATH = Path("results/final_dataset.csv")
OUTPUT_REPORT = Path("results/dataset_validation.txt")
# ----------------------------------------------------------

def validate_dataset():
    if not FINAL_DATA_PATH.exists():
        raise FileNotFoundError(f"{FINAL_DATA_PATH} not found!")

    df = pd.read_csv(FINAL_DATA_PATH, parse_dates=["date"])
    report_lines = []

    # 1. Basic info
    report_lines.append("=== Dataset Basic Info ===")
    report_lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    report_lines.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
    report_lines.append(f"Columns: {', '.join(df.columns)}\n")

    # 2. Missing values
    report_lines.append("=== Missing Values ===")
    missing = df.isnull().sum()
    report_lines.append(str(missing))
    report_lines.append("")

    # 3. NaN ratio for key columns
    key_cols = ['return', 'avg_sentiment', 'news_count']
    report_lines.append("=== NaN Ratio in Key Columns ===")
    for col in key_cols:
        if col in df.columns:
            nan_ratio = df[col].isnull().mean() * 100
            report_lines.append(f"{col}: {nan_ratio:.2f}% NaN")
    report_lines.append("")

    # 4. Descriptive statistics
    report_lines.append("=== Descriptive Statistics ===")
    report_lines.append(str(df.describe()))
    report_lines.append("")

    # 5. Check date continuity
    report_lines.append("=== Date Continuity Check ===")
    all_dates = pd.date_range(df['date'].min(), df['date'].max())
    missing_dates = all_dates.difference(df['date'].unique())
    if len(missing_dates) == 0:
        report_lines.append("All dates present (no missing days).")
    else:
        report_lines.append(f"Missing dates: {len(missing_dates)}")
        report_lines.append(str(missing_dates[:10]))  # preview first 10
    report_lines.append("")

    # 6. Abnormal values in return
    if 'return' in df.columns:
        report_lines.append("=== Abnormal Return Values (> +/- 50%) ===")
        abnormal = df[(df['return'] > 0.5) | (df['return'] < -0.5)]
        if abnormal.empty:
            report_lines.append("No abnormal returns found.")
        else:
            report_lines.append(f"Found {len(abnormal)} abnormal return rows.")
            report_lines.append(str(abnormal.head()))
        report_lines.append("")

    # Save full validation report
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Validation report saved to {OUTPUT_REPORT.resolve()}")
    print("Check dataset_validation.txt for details.")

    # 7. Generate summary report
    row_count = len(df)
    col_count = len(df.columns)
    date_min = df['date'].min()
    date_max = df['date'].max()
    missing_values = missing[missing > 0]
    nan_ratio_return = df['return'].isnull().mean() * 100 if 'return' in df.columns else 0
    nan_ratio_sent = df['avg_sentiment'].isnull().mean() * 100 if 'avg_sentiment' in df.columns else 0
    missing_dates_count = len(missing_dates)

    summary_path = OUTPUT_REPORT.parent / "validation_summary.txt"
    summary_content = [
        "Key Dataset Validation Results:",
        f"Rows: {row_count}, Columns: {col_count}",
        f"Date range: {date_min} to {date_max}",
        f"Missing Values:\n{missing_values}",
        f"NaN Ratio (return): {nan_ratio_return:.2f}%",
        f"NaN Ratio (avg_sentiment): {nan_ratio_sent:.2f}%",
        f"Missing Dates Count: {missing_dates_count}",
    ]
    summary_path.write_text("\n".join(summary_content), encoding="utf-8")
    print(f"Summary report saved to {summary_path.resolve()}")

if __name__ == "__main__":
    validate_dataset()
