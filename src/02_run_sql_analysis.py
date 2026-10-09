"""Step 2: Run every query in sql/analysis.sql, print and save results + a chart."""
import re
import sqlite3
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "churn.db"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

sql_text = (ROOT / "sql" / "analysis.sql").read_text()
blocks = re.split(r"-- name: (\w+)\n", sql_text)[1:]
queries = dict(zip(blocks[::2], blocks[1::2]))

results = {}
with sqlite3.connect(DB) as conn:
    for name, q in queries.items():
        results[name] = pd.read_sql_query(q, conn)
        results[name].to_csv(OUT / f"sql_{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(results[name].to_string(index=False))

d = results["churn_by_contract"]
plt.figure(figsize=(6, 4))
plt.bar(d["contract"], d["churn_rate_pct"], color=["#d9534f", "#f0ad4e", "#5cb85c"])
plt.ylabel("Churn rate (%)")
plt.title("Churn rate by contract type")
for i, v in enumerate(d["churn_rate_pct"]):
    plt.text(i, v + 0.5, f"{v}%", ha="center")
plt.tight_layout()
plt.savefig(OUT / "churn_by_contract.png", dpi=150)
print("\nSaved CSVs and chart to outputs/")
