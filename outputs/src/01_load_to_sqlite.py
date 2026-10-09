"""Step 1: Load the raw Telco churn CSV into a SQLite database (cleaned)."""
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "Telco-Customer-Churn.csv"
DB = ROOT / "data" / "churn.db"

df = pd.read_csv(CSV)

# TotalCharges has blank strings for brand-new customers -> convert to number
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)  # 1 = churned
df.columns = [c.lower() for c in df.columns]

with sqlite3.connect(DB) as conn:
    df.to_sql("customers", conn, if_exists="replace", index=False)
    n = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]

print(f"Loaded {n} rows into {DB}")
