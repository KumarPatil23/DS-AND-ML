import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA, PROCESSED_DATA, REPORTS
from utils import load_raw


df = load_raw(RAW_DATA)
initial_rows, initial_cols = df.shape
missing_before = int(df.isna().sum().sum())
duplicates_before = int(df.duplicated().sum())

numeric_cols = df.select_dtypes(include="number").columns
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include="object").columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode().iloc[0])

df = df.drop_duplicates().sort_values("timestamp").reset_index(drop=True)

# IQR flags are reported, not automatically deleted, so legitimate extreme pollution events are not lost.
outlier_rows = []
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        count = 0
    else:
        count = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
    outlier_rows.append((col, count))

out = pd.DataFrame(outlier_rows, columns=["column", "iqr_outlier_count"])
out.to_csv(REPORTS / "outlier_report.csv", index=False)
df.to_csv(PROCESSED_DATA, index=False)

summary = f"""EcoTwin Day 3 - Data Cleaning Summary\nInitial shape: {initial_rows} rows x {initial_cols} columns\nMissing values before cleaning: {missing_before}\nDuplicate rows before cleaning: {duplicates_before}\nFinal shape: {df.shape[0]} rows x {df.shape[1]} columns\nTimestamp parsing: successful\nOutliers: flagged in outputs/reports/outlier_report.csv; not deleted automatically.\nSaved: {PROCESSED_DATA}\n"""
(REPORTS / "day03_cleaning_summary.txt").write_text(summary, encoding="utf-8")
print(summary)
