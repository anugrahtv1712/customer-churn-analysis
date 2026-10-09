# Customer Churn Analysis & Prediction (SQL + Python)

An end-to-end project: load raw data into a **SQL database**, answer business questions with **SQL**, then train and compare **machine learning models** that predict which customers will leave, and save the best one for reuse.

**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), about 7,000 customers, 20 features.

## What it does
1. **Data engineering:** cleans the raw CSV and loads it into SQLite (`01_load_to_sqlite.py`).
2. **SQL analysis:** 7 queries using CTEs, `CASE`, aggregations and window functions (`sql/analysis.sql`) covering churn by contract, tenure, payment method, revenue lost, and a high-risk segment.
3. **Machine learning:** scikit-learn pipelines (one-hot encoding + scaling), compares Logistic Regression, Random Forest and Gradient Boosting with 5-fold cross-validation, evaluated on a held-out test set with ROC-AUC, precision, recall and F1.
4. **Reuse:** the best model is saved with joblib, and `predict.py` scores a new customer.

## Results
| Model | CV AUC | Test AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Gradient Boosting (best by AUC) | 0.847 | 0.844 | 0.664 | 0.529 | 0.589 |
| Logistic Regression | 0.846 | 0.842 | 0.504 | 0.783 | 0.614 |
| Random Forest | 0.844 | 0.842 | 0.555 | 0.738 | 0.634 |

All three models score almost the same on AUC. Gradient Boosting is the most precise, while Logistic Regression catches the most churners (78% recall). Which one to use depends on the cost of a missed churner versus a wasted retention offer.

### Key findings (from SQL analysis)
- Overall churn rate is **26.5%** (1,869 of 7,043 customers).
- Month-to-month customers churn at **42.7%**, versus **11.3%** (one year) and **2.8%** (two year).
- Customers in their first 12 months churn at **48.3%**, versus **9.6%** after 48 months.
- Electronic check users churn at **45.3%**, versus about 15-19% for other payment methods.
- High-risk segment (month-to-month + fiber optic + no tech support): **1,796 customers, 57.5% churn** vs 15.9% for everyone else.
- Month-to-month customers account for about **$120.8k of the monthly revenue lost** to churn (47% of that group's revenue).
- Top model drivers: month-to-month contract, short tenure, fiber optic internet, high monthly charges, no online security.

## How to run
```bash
pip install -r requirements.txt
# download the CSV from Kaggle and save it as data/Telco-Customer-Churn.csv
python src/01_load_to_sqlite.py
python src/02_run_sql_analysis.py
python src/03_train_model.py
python src/predict.py
```

## Structure
```
data/      raw CSV + SQLite database
sql/       analysis.sql (all queries)
src/       pipeline scripts
outputs/   charts, result tables, saved model
```

## Tech
Python, pandas, SQL (SQLite), scikit-learn, matplotlib, joblib
