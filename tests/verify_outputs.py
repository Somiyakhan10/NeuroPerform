"""Independent checks of NeuroPerform outputs.  Run: python tests/verify_outputs.py"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import (CLEAN_CSV, METRICS_FILE, MODEL_FEATURES, MODEL_FILE, RANDOM_STATE, TARGET,  # noqa: E402
                          engineer_grip_features, load_raw)

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


df = pd.read_csv(CLEAN_CSV)
metrics = json.loads(METRICS_FILE.read_text())
bundle = joblib.load(MODEL_FILE)
raw = load_raw()

# 1. Data integrity against the raw CDC files
check("Analytic sample size", len(df) == 2673, f"n = {len(df)}")
check("Participant IDs unique", df["participant_id"].is_unique)
check("All participants aged 60-80", df["age"].between(60, 80).all())
r = raw.set_index("SEQN").loc[df["participant_id"]]
check("Combined grip equals NHANES MGDCGSZ", np.allclose(df["grip_combined_kg"].values, r["MGDCGSZ"].values, atol=0.05, equal_nan=True),
      f"max diff {np.nanmax(np.abs(df['grip_combined_kg'].values - r['MGDCGSZ'].values)):.3f} kg")
check("DSST equals NHANES CFDDS", np.array_equal(df["dsst_score"].values, r["CFDDS"].values))
check("No NHANES missing codes left", df["tiredness_score"].dropna().between(0, 3).all()
      and df["education"].dropna().between(1, 5).all() and df["sleep_hours"].dropna().between(1, 24).all()
      and df["depressive_symptoms"].dropna().between(0, 24).all())
cut = df["dsst_score"].quantile(0.25)
check("Label = DSST below 25th percentile", (df[TARGET] == (df["dsst_score"] < cut)).all(), f"cut-off {cut:.0f}")

# 2. Feature engineering against a hand calculation
g = engineer_grip_features(np.array([[30, 32, 31]]), np.array([[24, 25, 26]]))
check("Grip combined (32 + 26 = 58)", np.isclose(g["grip_combined_kg"][0], 58))
check("Grip asymmetry (6/32 = 18.75%)", np.isclose(g["grip_asymmetry_pct"][0], 18.75))
cv = (np.std([30, 32, 31], ddof=1) / 31 + np.std([24, 25, 26], ddof=1) / 25) / 2 * 100
check("Grip trial CV", np.isclose(g["grip_cv_pct"][0], cv), f"{cv:.3f}%")
check("Grip change trial 1->3", np.isclose(g["grip_change_pct"][0], ((31 - 30) / 30 + (26 - 24) / 24) / 2 * 100))

# 3. Saved model reproduces the reported test metrics
X, y = df[MODEL_FEATURES], df[TARGET]
_, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
for name, model in bundle["all_models"].items():
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    m = metrics["models"][name]
    ok = all(np.isclose(v, m[k], atol=0.0015) for k, v in {
        "test_roc_auc": roc_auc_score(y_te, p), "test_accuracy": accuracy_score(y_te, pred),
        "test_precision": precision_score(y_te, pred), "test_recall": recall_score(y_te, pred),
        "test_f1": f1_score(y_te, pred)}.items())
    check(f"{name}: reported metrics reproduce", ok, f"AUC {roc_auc_score(y_te, p):.3f}")
check("Best model is highest CV AUC", metrics["best_model"] == max(metrics["models"], key=lambda k: metrics["models"][k]["cv_roc_auc_mean"]))

# 4. Dashboard input path gives the same prediction as the dataset row
row = df.dropna(subset=MODEL_FEATURES).iloc[0]
h1 = row[["mgxh1t1", "mgxh1t2", "mgxh1t3"]].values.astype(float)
h2 = row[["mgxh2t1", "mgxh2t2", "mgxh2t3"]].values.astype(float)
gg = engineer_grip_features(h1[None], h2[None])
x = {f: row[f] for f in MODEL_FEATURES}
x.update({k: gg[k][0] for k in ["grip_combined_kg", "grip_asymmetry_pct", "grip_cv_pct", "grip_change_pct"]})
x["grip_per_bmi"] = x["grip_combined_kg"] / x["bmi"]
p_app = bundle["model"].predict_proba(pd.DataFrame([x])[MODEL_FEATURES])[0, 1]
p_ref = bundle["model"].predict_proba(pd.DataFrame([row[MODEL_FEATURES]]).astype(float))[0, 1]
check("Dashboard feature path matches pipeline", np.isclose(p_app, p_ref, atol=1e-6), f"{p_app:.4f} vs {p_ref:.4f}")

# 5. Predictions move in plausible directions (Logistic Regression)
base = pd.DataFrame([{f: df[f].median() for f in MODEL_FEATURES}])
lr = bundle["all_models"]["Logistic Regression"]


def p_with(**kw):
    d = base.copy()
    for k, v in kw.items():
        d[k] = v
    return lr.predict_proba(d[MODEL_FEATURES])[0, 1]


check("Older age -> higher probability", p_with(age=80) > p_with(age=60))
check("Higher education -> lower probability", p_with(education=5) < p_with(education=1))
check("Stronger grip -> lower probability", p_with(grip_combined_kg=100, grip_per_bmi=100 / 28) < p_with(grip_combined_kg=35, grip_per_bmi=35 / 28))
check("Probabilities within [0, 1]", ((lr.predict_proba(X)[:, 1] >= 0) & (lr.predict_proba(X)[:, 1] <= 1)).all())

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
