"""Train the credit card default model. Run: python train.py  ->  writes model.pkl"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("UCI_Credit_Card.csv")
df = df.rename(columns={"default.payment.next.month": "default"}).drop(columns=["ID"])
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})   # unknown codes -> Others
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                  # unknown -> Others

X, y = df.drop(columns="default"), df["default"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("scale", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=2000, C=1.0))])

# Pick the decision threshold on out-of-fold predictions from the TRAIN split only,
# so the test set stays untouched.
oof = cross_val_predict(pipe, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
grid = np.arange(0.10, 0.60, 0.01)
threshold = float(grid[np.argmax([f1_score(y_tr, oof >= t) for t in grid])])

pipe.fit(X_tr, y_tr)
p = pipe.predict_proba(X_te)[:, 1]

def report(t):
    pred = (p >= t).astype(int)
    return {"threshold": round(t, 2),
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4)}

metrics = {"roc_auc": round(roc_auc_score(y_te, p), 4),
           "at_0.50": report(0.5), "at_tuned": report(threshold),
           "train_rows": len(X_tr), "test_rows": len(X_te)}
print(json.dumps(metrics, indent=2))

joblib.dump({"pipeline": pipe, "threshold": threshold,
             "features": list(X.columns), "metrics": metrics}, "model.pkl")
