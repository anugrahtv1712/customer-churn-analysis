-- name: overall_churn_rate
SELECT COUNT(*) AS customers,
       SUM(churn) AS churned,
       ROUND(100.0 * SUM(churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers;

-- name: churn_by_contract
SELECT contract,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 1) AS churn_rate_pct,
       ROUND(AVG(monthlycharges), 2) AS avg_monthly_charge
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- name: churn_by_tenure_bucket
WITH bucketed AS (
    SELECT *,
           CASE WHEN tenure < 12 THEN '1) 0-11 months'
                WHEN tenure < 24 THEN '2) 12-23 months'
                WHEN tenure < 48 THEN '3) 24-47 months'
                ELSE '4) 48+ months' END AS tenure_bucket
    FROM customers
)
SELECT tenure_bucket,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 1) AS churn_rate_pct
FROM bucketed
GROUP BY tenure_bucket
ORDER BY tenure_bucket;

-- name: churn_by_payment_method
SELECT paymentmethod,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 1) AS churn_rate_pct
FROM customers
GROUP BY paymentmethod
ORDER BY churn_rate_pct DESC;

-- name: revenue_lost_to_churn
SELECT contract,
       ROUND(SUM(CASE WHEN churn = 1 THEN monthlycharges ELSE 0 END), 0) AS monthly_revenue_lost,
       ROUND(100.0 * SUM(CASE WHEN churn = 1 THEN monthlycharges ELSE 0 END)
             / SUM(monthlycharges), 1) AS pct_of_contract_revenue
FROM customers
GROUP BY contract
ORDER BY monthly_revenue_lost DESC;

-- name: high_risk_segment
-- Month-to-month, fiber optic, no tech support vs everyone else
SELECT CASE WHEN contract = 'Month-to-month'
             AND internetservice = 'Fiber optic'
             AND techsupport = 'No' THEN 'High-risk segment' ELSE 'Everyone else' END AS segment,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 1) AS churn_rate_pct
FROM customers
GROUP BY segment;

-- name: top_charges_rank_within_contract
-- Window function: rank churned customers by monthly charge inside each contract type
SELECT * FROM (
    SELECT customerid, contract, monthlycharges,
           RANK() OVER (PARTITION BY contract ORDER BY monthlycharges DESC) AS charge_rank
    FROM customers
    WHERE churn = 1
)
WHERE charge_rank <= 3
ORDER BY contract, charge_rank;
