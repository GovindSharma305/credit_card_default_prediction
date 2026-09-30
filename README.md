# Credit Card Default Prediction

Flask web app that predicts whether a credit card client will default next month, and explains why.

- **Data:** UCI Credit Card Default dataset (30,000 clients, 23 features)
- **Model:** Logistic Regression in a scikit-learn Pipeline (StandardScaler + LogisticRegression)
- **Decision threshold:** tuned on out-of-fold predictions from the training split only
- **Explainability:** per-feature SHAP values (exact for a linear model) shown as the top 5 drivers of each prediction
- **API:** `POST /predict` (JSON), `GET /health`
- **Deploy:** Render, `gunicorn app:app`

## Results (held-out 20% test set, 6,000 rows)
| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.50 | 80.8% | 69.0% | 24.2% | 0.358 |
| 0.28 (tuned) | 78.7% | 52.0% | 48.8% | 0.503 |

ROC-AUC: 0.708

## Run locally
    pip install -r requirements.txt
    python train.py      # optional: retrains and rewrites model.pkl
    python app.py
