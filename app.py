import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__, static_folder="static", static_url_path="")

bundle = joblib.load("model.pkl")
pipe, THRESHOLD = bundle["pipeline"], bundle["threshold"]
FEATURES, METRICS = bundle["features"], bundle["metrics"]
scaler, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]

MONTHS = ["sept", "aug", "jul", "jun", "may", "apr"]          # newest -> oldest
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
EDU = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
MAR = {"Married": 1, "Single": 2, "Others": 3}

# field name in the form -> (dataset column, label shown in the UI)
FIELDS = {"limit_bal": ("LIMIT_BAL", "Credit limit"), "age": ("AGE", "Age")}
for i, m in enumerate(MONTHS):
    FIELDS[f"pay_status_{m}"] = (PAY_COLS[i], f"Repayment status ({m.title()})")
    FIELDS[f"bill_amt_{m}"] = (f"BILL_AMT{i+1}", f"Bill amount ({m.title()})")
    FIELDS[f"pay_amt_{m}"] = (f"PAY_AMT{i+1}", f"Payment made ({m.title()})")
LABELS = {col: label for col, label in FIELDS.values()}
LABELS.update({"SEX": "Sex", "EDUCATION": "Education", "MARRIAGE": "Marital status"})


def parse(form):
    """Validate the JSON body and return a one-row DataFrame in training column order."""
    row = {}
    for key, (col, label) in FIELDS.items():
        try:
            row[col] = float(form[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{label} is missing or not a number")
    if row["LIMIT_BAL"] <= 0:
        raise ValueError("Credit limit must be greater than 0")
    if not 18 <= row["AGE"] <= 100:
        raise ValueError("Age must be between 18 and 100")
    for col in PAY_COLS:
        if not -2 <= row[col] <= 9:
            raise ValueError("Repayment status must be between -2 and 9")
    if form.get("sex") not in ("Male", "Female"):
        raise ValueError("Select a sex")
    if form.get("education") not in EDU or form.get("marriage") not in MAR:
        raise ValueError("Select education and marital status")
    row["SEX"] = 1 if form["sex"] == "Male" else 2
    row["EDUCATION"] = EDU[form["education"]]
    row["MARRIAGE"] = MAR[form["marriage"]]
    return pd.DataFrame([row])[FEATURES]


def explain(df, top=5):
    """Per-feature contribution to the log-odds. For a linear model this is exactly
    the SHAP value (coefficient x standardised value, baseline = dataset mean)."""
    contrib = clf.coef_[0] * scaler.transform(df)[0]
    order = sorted(range(len(FEATURES)), key=lambda i: abs(contrib[i]), reverse=True)[:top]
    return [{"feature": LABELS.get(FEATURES[i], FEATURES[i]),
             "impact": round(float(contrib[i]), 3),
             "direction": "raises risk" if contrib[i] > 0 else "lowers risk"} for i in order]


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok", metrics=METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        df = parse(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    prob = float(pipe.predict_proba(df)[0, 1])
    flag = int(prob >= THRESHOLD)
    return jsonify(default_probability=prob, default_prediction=flag,
                   threshold=THRESHOLD, Result="High Risk" if flag else "Low Risk",
                   factors=explain(df))


if __name__ == "__main__":
    app.run(debug=True)
