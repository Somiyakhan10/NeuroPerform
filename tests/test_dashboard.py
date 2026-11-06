"""Dashboard tests: every page renders, and the live assessment updates correctly.  Run: python tests/test_dashboard.py"""

import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import MODEL_FEATURES, MODEL_FILE, engineer_grip_features  # noqa: E402

APP = str(ROOT / "app.py")
ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def shown_prob(at) -> float:
    html = next(m.value for m in at.markdown if 'class="np-result"' in m.value)
    return float(re.search(r'class="p">([\d.]+)%', html).group(1)) / 100


def expected(age, sex, edu, bmi, h1, h2, tired, phq8, sleep, trouble, vig, mod, mins, model="Logistic Regression"):
    g = {k: v[0] for k, v in engineer_grip_features(np.array([h1]), np.array([h2])).items()}
    x = {"age": age, "sex_female": int(sex == "Female"), "education": edu, "bmi": bmi, **{k: g[k] for k in
         ["grip_combined_kg", "grip_asymmetry_pct", "grip_cv_pct", "grip_change_pct"]}, "grip_per_bmi": g["grip_combined_kg"] / bmi,
         "tiredness_score": tired, "depressive_symptoms": phq8, "sleep_hours": sleep, "sleep_trouble": int(trouble),
         "vigorous_recreation": int(vig), "moderate_recreation": int(mod), "recreation_min_week": mins}
    m = joblib.load(MODEL_FILE)["all_models"][model]
    return float(m.predict_proba(pd.DataFrame([x])[MODEL_FEATURES])[0, 1])


# 1. Every page renders without exceptions
for page in ["Assessment", "Analytics", "Model", "Data"]:
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value(page).run()
    check(f"Page '{page}' renders", not at.exception, f"{len(at.get('plotly_chart'))} charts")

# 2. Live assessment: default profile matches a direct model calculation
at = AppTest.from_file(APP, default_timeout=180).run()
D = dict(age=68, sex="Male", edu=4, bmi=28.0, h1=[36, 37.5, 38], h2=[33, 34, 34.5], tired=1, phq8=2, sleep=7.0,
         trouble=False, vig=False, mod=True, mins=150)
p0 = shown_prob(at)
check("Default profile = direct model output", abs(p0 - expected(**D)) < 0.0006, f"{p0:.3f} vs {expected(**D):.3f}")

# 3. Changing inputs at runtime updates the output, in the expected direction
age_slider = next(s for s in at.slider if s.label == "Age")
age_slider.set_value(80).run()
p_age = shown_prob(at)
check("Age 68 -> 80 raises probability", p_age > p0, f"{p0:.3f} -> {p_age:.3f}")
check("Updated value = direct model output", abs(p_age - expected(**{**D, "age": 80})) < 0.0006)

edu_box = next(s for s in at.selectbox if s.label == "Education")
edu_box.set_value(1).run()
p_edu = shown_prob(at)
check("Education -> lowest raises probability", p_edu > p_age, f"{p_age:.3f} -> {p_edu:.3f}")

for i, v in enumerate([18, 18.5, 19]):
    next(n for n in at.number_input if n.label == f"R{i + 1}").set_value(v)
for i, v in enumerate([16, 16.5, 17]):
    next(n for n in at.number_input if n.label == f"L{i + 1}").set_value(v)
at.run()
p_weak = shown_prob(at)
check("Weaker grip raises probability", p_weak > p_edu, f"{p_edu:.3f} -> {p_weak:.3f}")
D2 = {**D, "age": 80, "edu": 1, "h1": [18, 18.5, 19], "h2": [16, 16.5, 17]}
check("Updated value = direct model output", abs(p_weak - expected(**D2)) < 0.0006, f"{p_weak:.3f}")
check("Classification label switches to Low", "Low processing speed" in next(m.value for m in at.markdown if 'class="np-result"' in m.value))

next(s for s in at.selectbox if s.label == "Model").set_value("Random Forest").run()
p_rf = shown_prob(at)
check("Switching model uses that model", abs(p_rf - expected(**D2, model="Random Forest")) < 0.0006, f"RF {p_rf:.3f}")
check("No exceptions during interaction", not at.exception)

# 4. Model page threshold slider updates metrics
at = AppTest.from_file(APP, default_timeout=180).run()
at.sidebar.radio[0].set_value("Model").run()
sens = lambda a: float(re.search(r'Sensitivity</div><div class="v">([\d.]+)', "".join(m.value for m in a.markdown)).group(1))
s50 = sens(at)
next(s for s in at.slider if s.label == "Decision threshold").set_value(0.3).run()
s30 = sens(at)
check("Lower threshold raises sensitivity", s30 > s50, f"{s50:.3f} -> {s30:.3f}")
check("Sensitivity at 0.50 matches saved recall", abs(s50 - 0.742) < 0.001)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
