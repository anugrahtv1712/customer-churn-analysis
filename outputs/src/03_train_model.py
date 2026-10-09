"""Step 3: Pull features with SQL, train + compare models, save the best one."""
import json
import sqlite3
from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

# ---- 1. Load data via SQL (with a small engineered feature) ----
query = """
SELECT gender, seniorcitizen, partner, dependents, tenure, phoneservice,
       multiplelines, internetservice, onlinesecurity, onlinebackup,
       deviceprotection, techsupport, streamingtv, streamingmovies,
       contract, paperlessbilling, paymentmethod, monthlycharges, totalcharges,
       monthlycharges * tenure AS expected_total,
       churn
FROM customers
"""
with sqlite3.connect(ROOT / "data" / "churn.db") as conn:
    df = pd.read_sql_query(query, conn)

X, y = df.drop(columns="churn"), df["churn"]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ---- 2. Preprocessing + models ----
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced", random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
rows, fitted = [], {}
for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("model", model)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({"model": name, "cv_auc": round(cv_auc, 3),
                 "test_auc": round(roc_auc_score(y_test, proba), 3),
                 "precision": round(precision_score(y_test, pred), 3),
                 "recall": round(recall_score(y_test, pred), 3),
                 "f1": round(f1_score(y_test, pred), 3)})
    fitted[name] = pipe

results = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
print(results.to_string(index=False))
results.to_csv(OUT / "model_comparison.csv", index=False)

# ---- 3. Best model: report, confusion matrix, feature importance ----
best_name = results.iloc[0]["model"]
best = fitted[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best.predict(X_test), target_names=["Stayed", "Churned"]))

ConfusionMatrixDisplay.from_estimator(best, X_test, y_test,
                                      display_labels=["Stayed", "Churned"], cmap="Blues")
plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close()

feat_names = best.named_steps["pre"].get_feature_names_out()
est = best.named_steps["model"]
imp = est.feature_importances_ if hasattr(est, "feature_importances_") else abs(est.coef_[0])
top = pd.Series(imp, index=feat_names).sort_values(ascending=False).head(12)[::-1]
top.plot(kind="barh", figsize=(7, 5), color="#337ab7")
plt.title(f"Top drivers of churn - {best_name}")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()

# ---- 4. Save model + metadata ----
joblib.dump(best, OUT / "churn_model.joblib")
(OUT / "metrics.json").write_text(json.dumps(
    {"best_model": best_name, **results.iloc[0].to_dict()}, indent=2, default=str))
print("Saved model, metrics and charts to outputs/")
