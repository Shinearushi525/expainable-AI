# 🛒 Return Risk AI

**Return Risk AI: Explainable E-Commerce Return Prediction & Decision Pipeline**

Predicts whether an e-commerce order will be returned, decides whether it is
**worth acting on** (cost-based), explains **why** in plain English, and serves
everything in an interactive Streamlit app backed by a BigQuery warehouse.

**Pipeline:** CSV → BigQuery (raw → stg → mart) → ML model → decision + explanation layer → Streamlit

## Key results (15,000 orders, 38.2% return rate)

| Model | ROC-AUC |
|---|---|
| **Gradient Boosting (best)** | **0.617** |
| Logistic Regression | 0.606 |
| Random Forest | 0.587 |
| XGBoost | 0.571 |
| Decision Tree | 0.535 |

- **Decision engine:** 4,638 of 15,000 orders (30.9%) are worth intervening on, for ≈ ₹80 lakh potential net savings.
- **Finding:** 0–10% discounts give the best expected profit (₹9,320/order); 50%+ discounts are returned 46.9% of the time.
- **Fairness caveat:** recall is 13.9% for Loyal vs 34.0% for New customers.

## App tabs

1. **Predict**: single order, risk gauge, business decision, "why this prediction?"
2. **Batch Prediction**: upload a CSV, get scores and decisions
3. **EDA Dashboard**: return reasons, KPIs, treemap, sunburst, correlation heatmap
4. **What-If Explorer**: sliders with live risk and recommendation
5. **Returns Map**: return rate by city
6. **BigQuery Pipeline**: how the data flows through the warehouse layers

## How the decision engine works

`expected_loss = P(return) × (₹150 + 15% × price)`. Act only if `expected_loss − ₹800 > 0`.
Explanations use a perturbation method (swap each feature to its median, measure the change in probability).

## Files

```
app.py                 Streamlit app
train_model.py         trains and compares 5 models, saves the best
decision_engine.py     cost-based intervention logic
explain.py             perturbation explainer
load_to_bigquery.py    CSV -> BigQuery raw.orders
transform.sql          raw -> stg -> mart (star schema)
predict.py             scores orders -> mart.fact_predictions
schema.sql             raw table definition
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The app also trains the model automatically if `best_model.pkl` is missing.
The BigQuery scripts (`load_to_bigquery.py`, `transform.sql`, `predict.py`) are the optional data-warehouse pipeline and are run separately with Google credentials.

## Statistical evaluation (`research_evaluation.py`)

- 5-fold CV x 2 repeats: Gradient Boosting ROC-AUC **0.622 ± 0.006** (majority baseline 0.500)
- Gradient Boosting is significantly better than every other model (corrected resampled t-test, p < 0.05)
- Ablation: removing logistics features hurts most (−0.034 AUC), then customer features (−0.025)
- Tuning gives only a small gain on held-out data (0.617 → 0.620)

Full tables are in `research_results/RESULTS.md`. Run with `python research_evaluation.py`.

## Tech stack

Python · scikit-learn · XGBoost · Streamlit · Plotly · Google BigQuery
