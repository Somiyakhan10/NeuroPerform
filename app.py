"""NeuroPerform | Cognitive-Motor Performance Analytics.  Run: streamlit run app.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pipeline import (CLEAN_CSV, FEATURE_LABELS, IMPORTANCE_FILE, METRICS_FILE, MODEL_FEATURES, MODEL_FILE,
                          RANDOM_STATE, TARGET, engineer_grip_features)

st.set_page_config(page_title="NeuroPerform", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Design system
# ----------------------------------------------------------------------------
INK, MUTED, LINE = "#1F2A24", "#66706A", "#E4DED2"
FOREST, SAGE, PALE, BEIGE, SAND, CLAY, GREY = "#3F5E4A", "#7FA087", "#DCE6DC", "#E3D6BC", "#B49A6E", "#A0583C", "#8E959B"
PALETTE = [FOREST, SAND, GREY, SAGE, CLAY, "#5E7C8C"]
SEQ = [[0, "#F3F0E8"], [0.35, "#C9D8C8"], [0.7, "#7FA087"], [1, "#2F4A3A"]]
DIV = [[0, "#A0583C"], [0.25, "#D2B48C"], [0.5, "#F7F5EF"], [0.75, "#9DB8A0"], [1, "#3F5E4A"]]
RISK = [[0, "#3F5E4A"], [0.3, "#8FAE92"], [0.5, "#EFE7D4"], [0.72, "#D2AE7E"], [1, "#A0583C"]]

pio.templates["neuro"] = go.layout.Template(layout=dict(
    font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=INK, size=12),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", colorway=PALETTE,
    title=dict(font=dict(size=14, color=INK), x=0.01, xanchor="left"),
    xaxis=dict(gridcolor="#EFEBE3", linecolor="#CFC8BA", zeroline=False, ticks="outside", tickcolor="#CFC8BA"),
    yaxis=dict(gridcolor="#EFEBE3", linecolor="#CFC8BA", zeroline=False, ticks="outside", tickcolor="#CFC8BA"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=LINE, font=dict(color=INK, family="IBM Plex Sans")),
    margin=dict(l=12, r=12, t=48, b=12)))
pio.templates.default = "simple_white+neuro"

SCENE_AXIS = dict(backgroundcolor="#F7F5F0", gridcolor="#E2DCCF", showbackground=True, zerolinecolor="#D5CEBF",
                  title_font=dict(size=11), tickfont=dict(size=10))
LIGHTING = dict(ambient=0.55, diffuse=0.85, specular=0.35, roughness=0.55, fresnel=0.25)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, p, label, input, textarea, button, select {{ font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif !important; }}
.stApp {{ background: radial-gradient(1200px 500px at 85% -10%, #EAF0E8 0%, rgba(234,240,232,0) 60%), #F7F5F0; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1320px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
footer {{ visibility: hidden; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #1F2D25 0%, #2A3B31 100%); border-right: none; }}
section[data-testid="stSidebar"] * {{ color: #E6EAE4; }}
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 4px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{ padding: 9px 14px; border-radius: 8px; width: 100%;
  transition: background .15s; margin: 0; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(255,255,255,.06); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{ background: rgba(143,174,146,.22);
  box-shadow: inset 3px 0 0 #9DB8A0; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] p {{ font-size: .95rem; font-weight: 500; }}

/* hero */
.np-hero {{ background: linear-gradient(120deg, #FFFFFF 0%, #F4F1EA 55%, #E6EEE4 100%); border: 1px solid {LINE};
  border-radius: 16px; padding: 22px 26px; margin-bottom: 18px; position: relative; overflow: hidden;
  box-shadow: 0 1px 2px rgba(31,42,36,.04), 0 10px 30px rgba(31,42,36,.05); }}
.np-hero:after {{ content: ""; position: absolute; right: -60px; top: -60px; width: 240px; height: 240px; border-radius: 50%;
  background: radial-gradient(circle, rgba(127,160,135,.25) 0%, rgba(127,160,135,0) 70%); }}
.np-eyebrow {{ font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; color: {FOREST}; font-weight: 600; }}
.np-title {{ font-size: 1.75rem; font-weight: 600; color: {INK}; margin: 2px 0 8px; letter-spacing: -.01em; }}
.np-chip {{ display: inline-block; font-size: .76rem; color: {FOREST}; background: rgba(127,160,135,.14);
  border: 1px solid rgba(63,94,74,.15); border-radius: 999px; padding: 3px 10px; margin: 0 6px 4px 0; }}

/* cards */
.np-kpi {{ background: #FFF; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 16px; position: relative;
  box-shadow: 0 1px 2px rgba(31,42,36,.04), 0 8px 24px rgba(31,42,36,.05); }}
.np-kpi:before {{ content: ""; position: absolute; left: 16px; right: 16px; top: 0; height: 3px; border-radius: 0 0 3px 3px;
  background: linear-gradient(90deg, {FOREST}, {SAGE}, {BEIGE}); }}
.np-kpi .l {{ font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; color: {MUTED}; margin-top: 4px; }}
.np-kpi .v {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem; color: {INK}; margin-top: 4px; }}
.np-kpi .s {{ font-size: .75rem; color: #8A918C; }}
.np-sec {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: {FOREST}; font-weight: 600;
  border-bottom: 1px solid {LINE}; padding-bottom: 6px; margin: 14px 0 10px; }}
.np-result {{ border-radius: 14px; padding: 18px; color: #FFF; height: 100%;
  box-shadow: 0 10px 30px rgba(31,42,36,.12); }}
.np-result .l {{ font-size: .7rem; letter-spacing: .14em; text-transform: uppercase; opacity: .8; }}
.np-result .v {{ font-size: 1.35rem; font-weight: 600; margin: 6px 0 2px; }}
.np-result .p {{ font-family: 'IBM Plex Mono', monospace; font-size: 2.3rem; line-height: 1.1; }}
.np-result .s {{ font-size: .78rem; opacity: .85; margin-top: 6px; }}

/* chart and table containers as cards */
div[data-testid="stPlotlyChart"] {{ background: #FFF; border: 1px solid {LINE}; border-radius: 14px; padding: 6px 6px 0;
  box-shadow: 0 1px 2px rgba(31,42,36,.04), 0 8px 24px rgba(31,42,36,.05); }}
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 12px; overflow: hidden; background: #FFF; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ background: transparent; border-radius: 8px 8px 0 0; padding: 8px 14px; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ background: #FFF; color: {INK}; border: 1px solid {LINE}; border-bottom-color: #FFF; }}
div[data-testid="stSlider"] div[role="slider"] {{ background: {FOREST}; }}
</style>
""", unsafe_allow_html=True)


def hero(eyebrow: str, title: str, chips: list[str]) -> None:
    st.markdown(f'<div class="np-hero"><div class="np-eyebrow">{eyebrow}</div><div class="np-title">{title}</div>'
                + "".join(f'<span class="np-chip">{c}</span>' for c in chips) + "</div>", unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="np-kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)


def sec(text: str, where=st) -> None:
    where.markdown(f'<div class="np-sec">{text}</div>', unsafe_allow_html=True)


def show(fig: go.Figure, height: int = 380, where=st) -> None:
    fig.update_layout(height=height)
    where.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def lab(f: str) -> str:
    return FEATURE_LABELS.get(f, f)


def pctile(s: pd.Series, v: float) -> float:
    s = s.dropna()
    return float(((s < v).mean() + 0.5 * (s == v).mean()) * 100)


# ----------------------------------------------------------------------------
# Data and models
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_CSV), json.loads(METRICS_FILE.read_text()), pd.read_csv(IMPORTANCE_FILE)


@st.cache_resource
def load_model():
    try:
        b = joblib.load(MODEL_FILE)
    except Exception:  # different scikit-learn version: retrain once
        from src.pipeline import run_pipeline
        run_pipeline(verbose=False)
        b = joblib.load(MODEL_FILE)
    for m in b.get("all_models", {}).values():  # single-threaded prediction is faster for interactive use
        if "n_jobs" in m.named_steps["model"].get_params():
            m.set_params(model__n_jobs=1)
    return b


if not (CLEAN_CSV.exists() and METRICS_FILE.exists() and MODEL_FILE.exists()):
    with st.spinner("Preparing data and models..."):
        from src.pipeline import run_pipeline
        run_pipeline(verbose=False)

df, metrics, imp = load_data()
bundle = load_model()
MODELS = bundle.get("all_models", {bundle["model_name"]: bundle["model"]})
best_name = metrics["best_model"]
best = metrics["models"][best_name]
CUTOFF = metrics["dsst_cutoff"]


@st.cache_data
def test_predictions():
    from sklearn.model_selection import train_test_split
    _, X_te, _, y_te = train_test_split(df[MODEL_FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET],
                                        random_state=RANDOM_STATE)
    return y_te.values, {n: m.predict_proba(X_te)[:, 1] for n, m in MODELS.items()}


COGNITIVE = ["dsst_score", "animal_fluency", "cerad_immediate", "cerad_delayed"]
MOTOR = ["grip_combined_kg", "grip_max_kg", "grip_asymmetry_pct", "grip_cv_pct", "grip_change_pct", "grip_per_bmi"]
FATIGUE = ["tiredness_score", "depressive_symptoms", "sleep_hours", "sleep_trouble"]
ACTIVITY = ["vigorous_recreation", "moderate_recreation", "recreation_min_week"]
DEMO = ["age", "sex_female", "education", "bmi"]
GROUPS = {"Cognitive": COGNITIVE, "Motor": MOTOR, "Fatigue & sleep": FATIGUE, "Activity": ACTIVITY, "Demographics": DEMO}
NUMERIC = COGNITIVE + MOTOR + FATIGUE + ACTIVITY + DEMO
EDU = {1: "Less than 9th grade", 2: "9-11th grade", 3: "High school / GED", 4: "Some college / AA", 5: "College graduate +"}
TIRED = {0: "Not at all", 1: "Several days", 2: "More than half the days", 3: "Nearly every day"}

dfp = df.copy()
dfp["Sex"] = dfp["sex_female"].map({0: "Male", 1: "Female"})
dfp["Age group"] = pd.cut(dfp["age"], [59, 64, 69, 74, 80], labels=["60-64", "65-69", "70-74", "75-80"]).astype(str)
dfp["Tiredness"] = dfp["tiredness_score"].map(TIRED)
dfp["Processing speed"] = dfp["low_processing_speed"].map({1: "Low", 0: "Typical"})
dfp["Education"] = dfp["education"].map(EDU)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin:6px 0 22px">
  <svg width="34" height="34" viewBox="0 0 34 34"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
  <stop offset="0" stop-color="#9DB8A0"/><stop offset="1" stop-color="#E3D6BC"/></linearGradient></defs>
  <circle cx="17" cy="17" r="15" fill="none" stroke="url(#g)" stroke-width="2"/>
  <path d="M7 18 L12 18 L14.5 11 L18 24 L21 14 L23 18 L27 18" fill="none" stroke="#E3D6BC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
  <div><div style="font-size:1.2rem;font-weight:700;letter-spacing:.01em">NeuroPerform</div>
  <div style="font-size:.74rem;opacity:.7">Cognitive-Motor Analytics</div></div></div>""", unsafe_allow_html=True)
    page = st.radio("Navigation", ["Assessment", "Analytics", "Model", "Data"], label_visibility="collapsed")
    st.markdown(f"""
<div style="margin-top:28px;padding:12px 14px;border:1px solid rgba(255,255,255,.1);border-radius:10px;font-size:.76rem;line-height:1.7">
<span style="opacity:.6">Cohort</span><br><b>{len(df):,}</b> adults · NHANES 2011-14<br>
<span style="opacity:.6">Model</span><br><b>{best_name}</b> · AUC {best['test_roc_auc']:.3f}</div>
<div style="font-size:.7rem;opacity:.5;margin-top:14px">Experimental. Not a diagnostic tool.</div>""", unsafe_allow_html=True)


# ============================================================================
# ASSESSMENT (live)
# ============================================================================
def build_x(age, sex, edu, bmi, h1, h2, tired, phq8, sleep, trouble, vig, mod, mins) -> dict:
    g = {k: float(v[0]) for k, v in engineer_grip_features(np.array([h1]), np.array([h2])).items()}
    return {"age": age, "sex_female": int(sex == "Female"), "education": edu, "bmi": bmi,
            "grip_combined_kg": g["grip_combined_kg"], "grip_asymmetry_pct": g["grip_asymmetry_pct"],
            "grip_cv_pct": g["grip_cv_pct"], "grip_change_pct": g["grip_change_pct"],
            "grip_per_bmi": g["grip_combined_kg"] / bmi, "tiredness_score": tired, "depressive_symptoms": phq8,
            "sleep_hours": sleep, "sleep_trouble": int(trouble), "vigorous_recreation": int(vig),
            "moderate_recreation": int(mod), "recreation_min_week": mins}


def predict(model, rows: list[dict]) -> np.ndarray:
    return model.predict_proba(pd.DataFrame(rows)[MODEL_FEATURES])[:, 1]


if page == "Assessment":
    hero("Live assessment", "Cognitive-Motor Risk Profile",
         ["Updates instantly", f"{len(MODELS)} trained models", "3D risk surface", "Sensitivity analysis"])

    left, right = st.columns([1, 2.1], gap="large")
    with left:
        sec("Profile", left)
        c = st.columns(2)
        age = c[0].slider("Age", 60, 80, 68)
        bmi = c[1].slider("BMI", 16.0, 50.0, 28.0, 0.5)
        c = st.columns(2)
        sex = c[0].selectbox("Sex", ["Male", "Female"])
        edu = c[1].selectbox("Education", list(EDU), index=3, format_func=lambda k: EDU[k])

        sec("Grip dynamometry (kg)", left)
        c = st.columns(3)
        h1 = [c[i].number_input(f"R{i + 1}", 2.0, 90.0, [36.0, 37.5, 38.0][i], 0.5) for i in range(3)]
        c = st.columns(3)
        h2 = [c[i].number_input(f"L{i + 1}", 2.0, 90.0, [33.0, 34.0, 34.5][i], 0.5) for i in range(3)]

        sec("Fatigue & sleep", left)
        tired = st.select_slider("Tiredness (past 2 weeks)", options=list(TIRED), value=1, format_func=lambda k: TIRED[k])
        c = st.columns(2)
        phq8 = c[0].slider("Mood (PHQ-8)", 0, 24, 2)
        sleep = c[1].slider("Sleep (h)", 3.0, 11.0, 7.0, 0.5)
        trouble = st.toggle("Diagnosed sleep trouble", value=False)

        sec("Activity", left)
        mins = st.slider("Recreational activity (min/week)", 0, 900, 150, 10)
        c = st.columns(2)
        vig = c[0].toggle("Vigorous", value=False)
        mod = c[1].toggle("Moderate", value=mins > 0)

        sec("Model", left)
        model_name = st.selectbox("Model", list(MODELS), index=list(MODELS).index(bundle["model_name"]), label_visibility="collapsed")

    model = MODELS[model_name]
    x = build_x(age, sex, edu, bmi, h1, h2, tired, phq8, sleep, trouble, vig, mod, mins)
    prob = float(predict(model, [x])[0])
    low = prob >= 0.5
    same_sex = df[df["sex_female"] == x["sex_female"]]

    with right:
        a, b, c_ = st.columns([1.05, 1, 1.15])
        grad = "linear-gradient(135deg,#8E4A31 0%,#B9774F 100%)" if low else "linear-gradient(135deg,#2F4A3A 0%,#5E8268 100%)"
        a.markdown(f'<div class="np-result" style="background:{grad}"><div class="l">Classification</div>'
                   f'<div class="v">{"Low processing speed" if low else "Typical processing speed"}</div>'
                   f'<div class="p">{prob * 100:.1f}%</div><div class="s">probability · {model_name}</div>'
                   f'<div class="s">Cohort base rate {df[TARGET].mean() * 100:.1f}%</div></div>', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 30, "family": "IBM Plex Mono"}},
            gauge={"axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CFC8BA"}, "bar": {"color": INK, "thickness": .18},
                   "bgcolor": "white", "borderwidth": 0,
                   "steps": [{"range": [i * 10, i * 10 + 10], "color": px.colors.sample_colorscale(RISK, i / 9)[0]} for i in range(10)],
                   "threshold": {"line": {"color": INK, "width": 3}, "thickness": .9, "value": 50}}))
        fig.update_layout(margin=dict(l=24, r=24, t=36, b=6), title="Risk gauge")
        show(fig, 215, b)

        radar = {"Strength": pctile(same_sex["grip_combined_kg"], x["grip_combined_kg"]),
                 "Symmetry": 100 - pctile(df["grip_asymmetry_pct"], x["grip_asymmetry_pct"]),
                 "Consistency": 100 - pctile(df["grip_cv_pct"], x["grip_cv_pct"]),
                 "Activity": pctile(df["recreation_min_week"], x["recreation_min_week"]),
                 "Mood": 100 - pctile(df["depressive_symptoms"], x["depressive_symptoms"]),
                 "Energy": 100 - pctile(df["tiredness_score"], x["tiredness_score"])}
        th = list(radar) + [list(radar)[0]]
        r_ = list(radar.values()) + [list(radar.values())[0]]
        fig = go.Figure()
        fig.add_scatterpolar(r=[50] * len(th), theta=th, line=dict(color=GREY, dash="dot", width=1), name="Cohort median", hoverinfo="skip")
        fig.add_scatterpolar(r=r_, theta=th, fill="toself", fillcolor="rgba(63,94,74,.22)", line=dict(color=FOREST, width=2),
                             name="Profile", hovertemplate="%{theta}: %{r:.0f}th percentile<extra></extra>")
        fig.update_layout(title="Percentile profile", showlegend=False, margin=dict(l=52, r=52, t=44, b=22),
                          polar=dict(bgcolor="#FBFAF6", radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75], tickfont=dict(size=8),
                                     gridcolor="#E8E2D6", showticklabels=False),
                                     angularaxis=dict(gridcolor="#E8E2D6", tickfont=dict(size=9.5))))
        show(fig, 215, c_)

        # 3D risk surface: probability over age x combined grip, all other inputs held at the current profile
        ages = np.arange(60, 81, 1)
        grips = np.linspace(max(15, df["grip_combined_kg"].quantile(.01)), df["grip_combined_kg"].quantile(.99), 32)
        rows = []
        for gk in grips:
            for ag in ages:
                r = dict(x, age=ag, grip_combined_kg=gk, grip_per_bmi=gk / bmi)
                rows.append(r)
        Z = predict(model, rows).reshape(len(grips), len(ages))
        fig = go.Figure()
        fig.add_surface(x=ages, y=grips, z=Z, colorscale=RISK, cmin=0, cmax=1, opacity=.97, lighting=LIGHTING,
                        lightposition=dict(x=100, y=200, z=1000),
                        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1, start=.1, end=.9, size=.1)),
                        colorbar=dict(title="P(low)", thickness=12, len=.6, tickformat=".0%"),
                        hovertemplate="Age %{x}<br>Grip %{y:.0f} kg<br>P = %{z:.1%}<extra></extra>")
        fig.add_scatter3d(x=[age], y=[x["grip_combined_kg"]], z=[prob + .02], mode="markers+text",
                          marker=dict(size=7, color="#FFFFFF", line=dict(color=INK, width=3)), text=["Profile"],
                          textposition="top center", textfont=dict(color=INK, size=11), showlegend=False,
                          hovertemplate=f"Current profile<br>P = {prob:.1%}<extra></extra>")
        fig.update_layout(title="3D risk surface · age × grip strength", margin=dict(l=0, r=0, t=40, b=0),
                          scene=dict(xaxis=dict(SCENE_AXIS, title="Age"), yaxis=dict(SCENE_AXIS, title="Combined grip (kg)"),
                                     zaxis=dict(SCENE_AXIS, title="Probability", range=[0, 1], tickformat=".0%"),
                                     camera=dict(eye=dict(x=1.2, y=-1.0, z=.78), center=dict(x=0, y=0, z=-.15)),
                                     aspectratio=dict(x=1, y=1.15, z=.9)))
        show(fig, 500)

        # Local sensitivity: change each input by ±1 cohort SD (±1 level for ordinal inputs)
        steps = {"age": df["age"].std(), "education": 1, "bmi": df["bmi"].std(),
                 "grip_combined_kg": df["grip_combined_kg"].std(), "grip_asymmetry_pct": df["grip_asymmetry_pct"].std(),
                 "grip_cv_pct": df["grip_cv_pct"].std(), "grip_change_pct": df["grip_change_pct"].std(),
                 "tiredness_score": 1, "depressive_symptoms": df["depressive_symptoms"].std(),
                 "sleep_hours": df["sleep_hours"].std(), "recreation_min_week": df["recreation_min_week"].std()}
        R = metrics["feature_ranges"]
        srows = []
        for f, s in steps.items():
            for sign in (-1, 1):
                v = float(np.clip(x[f] + sign * s, R[f]["min"], R[f]["max"]))
                r = dict(x, **{f: v})
                if f in ("grip_combined_kg", "bmi"):
                    r["grip_per_bmi"] = r["grip_combined_kg"] / r["bmi"]
                srows.append(r)
        sp = predict(model, srows).reshape(-1, 2) - prob
        sens = pd.DataFrame({"feature": [lab(f) for f in steps], "lo": sp[:, 0] * 100, "hi": sp[:, 1] * 100})
        sens["span"] = sens[["lo", "hi"]].abs().max(axis=1)
        sens = sens.sort_values("span")
        fig = go.Figure()
        fig.add_bar(y=sens["feature"], x=sens["lo"], orientation="h", name="−1 SD", marker=dict(color=SAND, line=dict(width=0)),
                    hovertemplate="%{y}<br>−1 SD: %{x:+.1f} pp<extra></extra>")
        fig.add_bar(y=sens["feature"], x=sens["hi"], orientation="h", name="+1 SD", marker=dict(color=FOREST, line=dict(width=0)),
                    hovertemplate="%{y}<br>+1 SD: %{x:+.1f} pp<extra></extra>")
        fig.add_vline(x=0, line_color=INK, line_width=1)
        fig.update_layout(title="Sensitivity · change in probability (percentage points)", barmode="overlay",
                          legend=dict(orientation="h", y=1.08, x=.6), xaxis_title="Δ probability (pp)")
        show(fig, 400)

        out = [lab(f) for f in MODEL_FEATURES if not (R[f]["min"] <= x[f] <= R[f]["max"])]
        if out:
            st.warning("Outside training range: " + ", ".join(out))


# ============================================================================
# ANALYTICS
# ============================================================================
elif page == "Analytics":
    hero("Cohort analytics", "Cognitive-Motor Patterns",
         [f"n = {len(df):,}", "Spearman / OLS", "3D response surface", "Interactive"])
    tabs = st.tabs(["3D response surface", "3D cohort", "Processing speed", "Motor", "Fatigue", "Correlations", "Compare"])

    with tabs[0]:
        c = st.columns([1, 1, 1])
        zvar = c[0].selectbox("Outcome", COGNITIVE, format_func=lab)
        sex_f = c[1].radio("Sex", ["All", "Male", "Female"], horizontal=True)
        show_pts = c[2].toggle("Show participants", value=True)
        d = dfp if sex_f == "All" else dfp[dfp["Sex"] == sex_f]
        d = d.dropna(subset=["age", "grip_combined_kg", zvar])
        a_, g_, z_ = d["age"].values, d["grip_combined_kg"].values, d[zvar].values
        A = np.column_stack([np.ones_like(a_), a_, g_, a_ ** 2, g_ ** 2, a_ * g_])
        beta, *_ = np.linalg.lstsq(A, z_, rcond=None)
        r2 = 1 - ((z_ - A @ beta) ** 2).sum() / ((z_ - z_.mean()) ** 2).sum()
        ag = np.linspace(60, 80, 30)
        gg = np.linspace(np.percentile(g_, 2), np.percentile(g_, 98), 30)
        AG, GG = np.meshgrid(ag, gg)
        ZZ = beta[0] + beta[1] * AG + beta[2] * GG + beta[3] * AG ** 2 + beta[4] * GG ** 2 + beta[5] * AG * GG
        fig = go.Figure()
        fig.add_surface(x=ag, y=gg, z=ZZ, colorscale=SEQ, opacity=.92, lighting=LIGHTING, lightposition=dict(x=100, y=200, z=1000),
                        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
                        colorbar=dict(title=lab(zvar).split(" (")[0], thickness=12, len=.6),
                        hovertemplate="Age %{x:.0f}<br>Grip %{y:.0f} kg<br>Fitted %{z:.1f}<extra></extra>")
        if show_pts:
            fig.add_scatter3d(x=a_, y=g_, z=z_, mode="markers", marker=dict(size=2.2, color=z_, colorscale=SEQ, opacity=.55),
                              showlegend=False, hovertemplate="Age %{x}<br>Grip %{y:.0f}<br>Score %{z}<extra></extra>")
        fig.update_layout(title=f"Quadratic response surface · {lab(zvar)}  (R² = {r2:.3f}, n = {len(d):,})",
                          margin=dict(l=0, r=0, t=40, b=0),
                          scene=dict(xaxis=dict(SCENE_AXIS, title="Age"), yaxis=dict(SCENE_AXIS, title="Combined grip (kg)"),
                                     zaxis=dict(SCENE_AXIS, title=lab(zvar).split(" (")[0]),
                                     camera=dict(eye=dict(x=1.3, y=-1.15, z=.7)), aspectratio=dict(x=1, y=1.1, z=.75)))
        show(fig, 620)

    with tabs[1]:
        c = st.columns(4)
        xv = c[0].selectbox("X", NUMERIC, index=NUMERIC.index("age"), format_func=lab, key="x3")
        yv = c[1].selectbox("Y", NUMERIC, index=NUMERIC.index("grip_combined_kg"), format_func=lab, key="y3")
        zv = c[2].selectbox("Z", NUMERIC, index=NUMERIC.index("dsst_score"), format_func=lab, key="z3")
        cv = c[3].selectbox("Colour", ["dsst_score", "Processing speed", "Sex", "Age group", "Education"],
                            format_func=lambda v: lab(v) if v in NUMERIC else v)
        d = dfp.dropna(subset=[xv, yv, zv])
        if cv in NUMERIC:
            fig = px.scatter_3d(d, x=xv, y=yv, z=zv, color=cv, color_continuous_scale=SEQ, opacity=.75,
                                labels={xv: lab(xv), yv: lab(yv), zv: lab(zv), cv: lab(cv)})
        else:
            fig = px.scatter_3d(d, x=xv, y=yv, z=zv, color=cv, color_discrete_sequence=PALETTE, opacity=.7,
                                labels={xv: lab(xv), yv: lab(yv), zv: lab(zv)})
        fig.update_traces(marker=dict(size=2.8, line=dict(width=0)))
        fig.update_layout(title="Cohort in 3D feature space", margin=dict(l=0, r=0, t=40, b=0),
                          scene=dict(xaxis=SCENE_AXIS, yaxis=SCENE_AXIS, zaxis=SCENE_AXIS, camera=dict(eye=dict(x=1.3, y=-1.2, z=.75))))
        show(fig, 600)

    with tabs[2]:
        a, b = st.columns([3, 2])
        split = a.radio("Colour by", ["None", "Sex", "Age group", "Education"], horizontal=True)
        fig = px.histogram(dfp, x="dsst_score", color=None if split == "None" else split, nbins=40, barmode="overlay",
                           opacity=.78, labels={"dsst_score": "DSST score"}, color_discrete_sequence=PALETTE, marginal="box")
        fig.add_vline(x=CUTOFF, line_dash="dash", line_color=CLAY, annotation_text=f"cut-off {CUTOFF:.0f}", annotation_font_color=CLAY)
        fig.update_layout(title="DSST distribution", bargap=.04)
        show(fig, 420, a)
        fig = px.violin(dfp.sort_values("Age group"), x="Age group", y="dsst_score", color="Sex", box=True, points=False,
                        labels={"dsst_score": "DSST score"}, color_discrete_sequence=[SAND, FOREST])
        fig.update_layout(title="DSST by age and sex", violinmode="group")
        b.write("")
        show(fig, 420, b)
        cog = st.radio("Measure", ["animal_fluency", "cerad_immediate", "cerad_delayed"], format_func=lab, horizontal=True)
        agg = dfp.groupby(["Age group", "Sex"])[cog].agg(["mean", "sem"]).reset_index()
        fig = px.line(agg, x="Age group", y="mean", color="Sex", error_y="sem", markers=True, color_discrete_sequence=[SAND, FOREST],
                      labels={"mean": lab(cog)})
        fig.update_layout(title=f"{lab(cog)} · mean ± SEM")
        show(fig, 340)

    with tabs[3]:
        a, b = st.columns(2)
        trial = pd.DataFrame([{"Trial": f"T{t}", "Hand": f"Hand {h}", "mean": df[f"mgxh{h}t{t}"].mean(), "sem": df[f"mgxh{h}t{t}"].sem()}
                              for h in (1, 2) for t in (1, 2, 3)])
        fig = px.line(trial, x="Trial", y="mean", color="Hand", error_y="sem", markers=True, color_discrete_sequence=[FOREST, SAND],
                      labels={"mean": "Grip (kg)"})
        fig.update_layout(title="Force across repeated maximal efforts · mean ± SEM")
        show(fig, 380, a)
        fig = px.histogram(dfp, x="grip_change_pct", color="Sex", nbins=70, barmode="overlay", opacity=.75,
                           color_discrete_sequence=[SAND, FOREST], labels={"grip_change_pct": "Force change T1 → T3 (%)"})
        fig.add_vline(x=0, line_dash="dash", line_color=INK)
        fig.update_layout(title="Repeated-effort force change", bargap=.03)
        show(fig, 380, b)
        c1, c2 = st.columns(2)
        tx = c1.selectbox("X", ["age", "bmi", "recreation_min_week", "depressive_symptoms"], format_func=lab)
        my = c2.selectbox("Motor variable", MOTOR, format_func=lab)
        fig = px.density_contour(dfp, x=tx, y=my, color="Sex", color_discrete_sequence=[SAND, FOREST],
                                 labels={tx: lab(tx), my: lab(my)}, marginal_x="histogram", marginal_y="histogram")
        fig.update_layout(title=f"{lab(my)} vs {lab(tx)} · density")
        show(fig, 460)

    with tabs[4]:
        order = list(TIRED.values())
        d = dfp.dropna(subset=["Tiredness"])
        a, b = st.columns(2)
        fig = px.box(d, x="Tiredness", y="dsst_score", color="Sex", category_orders={"Tiredness": order},
                     color_discrete_sequence=[SAND, FOREST], labels={"dsst_score": "DSST score"}, points=False)
        fig.update_layout(title="DSST by tiredness")
        show(fig, 400, a)
        fig = px.box(d, x="Tiredness", y="grip_combined_kg", color="Sex", category_orders={"Tiredness": order},
                     color_discrete_sequence=[SAND, FOREST], labels={"grip_combined_kg": "Combined grip (kg)"}, points=False)
        fig.update_layout(title="Grip by tiredness")
        show(fig, 400, b)
        sl = dfp.dropna(subset=["sleep_hours"]).copy()
        sl["Sleep (h)"] = sl["sleep_hours"].clip(4, 10).round().astype(int)
        hm = sl.pivot_table(index="Tiredness", columns="Sleep (h)", values="dsst_score", aggfunc="mean").reindex(order)
        fig = px.imshow(hm, text_auto=".1f", color_continuous_scale=SEQ, aspect="auto", labels=dict(color="Mean DSST"))
        fig.update_layout(title="Mean DSST · tiredness × sleep duration (≤4 and ≥10 h pooled)")
        show(fig, 360)

    with tabs[5]:
        sel = st.multiselect("Variables", NUMERIC, format_func=lab,
                             default=["dsst_score", "animal_fluency", "cerad_immediate", "cerad_delayed", "grip_combined_kg",
                                      "grip_cv_pct", "grip_asymmetry_pct", "grip_change_pct", "tiredness_score",
                                      "sleep_hours", "recreation_min_week", "age", "education"])
        method = st.radio("Method", ["spearman", "pearson"], horizontal=True)
        if len(sel) >= 2:
            cm = df[sel].corr(method=method)
            mask = np.triu(np.ones_like(cm, dtype=bool), k=1)
            fig = px.imshow(cm.mask(mask), x=[lab(s) for s in sel], y=[lab(s) for s in sel], zmin=-1, zmax=1,
                            color_continuous_scale=DIV, text_auto=".2f", aspect="auto")
            fig.update_traces(textfont_size=10)
            fig.update_layout(title=f"{method.title()} correlation matrix", plot_bgcolor="rgba(0,0,0,0)")
            fig.update_xaxes(showgrid=False, tickangle=-35)
            fig.update_yaxes(showgrid=False)
            show(fig, 640)

    with tabs[6]:
        a, b, c_ = st.columns(3)
        xv = a.selectbox("X axis", NUMERIC, index=NUMERIC.index("grip_combined_kg"), format_func=lab)
        yv = b.selectbox("Y axis", NUMERIC, index=NUMERIC.index("dsst_score"), format_func=lab)
        col = c_.selectbox("Colour", ["Sex", "Age group", "Processing speed", "Education"])
        fig = px.scatter(dfp, x=xv, y=yv, color=col, opacity=.55, trendline="ols", trendline_scope="overall",
                         trendline_color_override=INK, labels={xv: lab(xv), yv: lab(yv)}, color_discrete_sequence=PALETTE,
                         marginal_x="box", marginal_y="box")
        fig.update_traces(marker=dict(size=5, line=dict(width=0)), selector=dict(mode="markers"))
        if xv != yv:
            from scipy.stats import spearmanr
            sub = df[[xv, yv]].dropna()
            r, p = spearmanr(sub[xv], sub[yv])
            fig.update_layout(title=f"Spearman ρ = {r:.3f} · {'p < 0.001' if p < 0.001 else f'p = {p:.3f}'} · n = {len(sub):,}")
        show(fig, 540)


# ============================================================================
# MODEL
# ============================================================================
elif page == "Model":
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score, precision_recall_curve, precision_score,
                                 recall_score, roc_curve, average_precision_score)

    hero("Model evaluation", "Classifier Performance",
         [f"Train {metrics['n_train']:,} / test {metrics['n_test']:,}", "5-fold stratified CV", "Adjustable threshold", "Calibration"])
    y_te, probs = test_predictions()

    c = st.columns([1.2, 2])
    model_sel = c[0].selectbox("Model", list(MODELS), index=list(MODELS).index(best_name))
    thr = c[1].slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    p = probs[model_sel]
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    k = st.columns(6)
    kpi(k[0], "ROC-AUC", f"{metrics['models'][model_sel]['test_roc_auc']:.3f}", f"CV {metrics['models'][model_sel]['cv_roc_auc_mean']:.3f}")
    kpi(k[1], "Sensitivity", f"{recall_score(y_te, pred):.3f}", "recall")
    kpi(k[2], "Specificity", f"{tn / (tn + fp):.3f}", "true-negative rate")
    kpi(k[3], "Precision", f"{precision_score(y_te, pred, zero_division=0):.3f}", "PPV")
    kpi(k[4], "F1-score", f"{f1_score(y_te, pred):.3f}", f"threshold {thr:.2f}")
    kpi(k[5], "Brier score", f"{brier_score_loss(y_te, p):.3f}", "lower is better")
    st.write("")

    a, b, c_ = st.columns(3)
    cmx = np.array([[tn, fp], [fn, tp]])
    pct = cmx / cmx.sum(axis=1, keepdims=True) * 100
    fig = go.Figure(go.Heatmap(z=pct, x=["Typical", "Low"], y=["Typical", "Low"], colorscale=SEQ, showscale=False, zmin=0, zmax=100,
                               text=[[f"<b>{cmx[i, j]}</b><br>{pct[i, j]:.0f}%" for j in range(2)] for i in range(2)],
                               texttemplate="%{text}", textfont=dict(size=15), xgap=3, ygap=3))
    fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    show(fig, 360, a)

    fig = go.Figure()
    for n, colr in zip(MODELS, [FOREST, SAND, GREY]):
        fpr, tpr, _ = roc_curve(y_te, probs[n])
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=n, line=dict(color=colr, width=2.4 if n == model_sel else 1.4),
                        opacity=1 if n == model_sel else .55)
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#C9C3B6"), showlegend=False)
    fig.add_scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode="markers", marker=dict(size=11, color=CLAY, line=dict(color="white", width=2)),
                    name="Operating point")
    fig.update_layout(title="ROC curve", xaxis_title="1 − specificity", yaxis_title="Sensitivity", legend=dict(x=.42, y=.05, font=dict(size=10)))
    show(fig, 360, b)

    pr, rc, _ = precision_recall_curve(y_te, p)
    fig = go.Figure()
    fig.add_scatter(x=rc, y=pr, mode="lines", line=dict(color=FOREST, width=2.4), fill="tozeroy", fillcolor="rgba(63,94,74,.08)",
                    name=f"AP = {average_precision_score(y_te, p):.3f}")
    fig.add_hline(y=y_te.mean(), line_dash="dot", line_color=GREY, annotation_text="base rate", annotation_font_color=GREY)
    fig.add_scatter(x=[recall_score(y_te, pred)], y=[precision_score(y_te, pred, zero_division=0)], mode="markers",
                    marker=dict(size=11, color=CLAY, line=dict(color="white", width=2)), name="Operating point")
    fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 1.02],
                      legend=dict(x=.45, y=.98, font=dict(size=10)))
    show(fig, 360, c_)

    a, b = st.columns(2)
    ths = np.linspace(.05, .95, 91)
    curves = pd.DataFrame({"Threshold": ths,
                           "Sensitivity": [recall_score(y_te, p >= t) for t in ths],
                           "Specificity": [((p < t) & (y_te == 0)).sum() / (y_te == 0).sum() for t in ths],
                           "Precision": [precision_score(y_te, p >= t, zero_division=0) for t in ths],
                           "F1": [f1_score(y_te, p >= t) for t in ths]})
    fig = px.line(curves, x="Threshold", y=["Sensitivity", "Specificity", "Precision", "F1"],
                  color_discrete_sequence=[FOREST, SAND, GREY, CLAY])
    fig.add_vline(x=thr, line_color=INK, line_dash="dash")
    fig.update_layout(title="Metric trade-off across thresholds", yaxis_title="", legend_title=None, legend=dict(orientation="h", y=-.2))
    show(fig, 380, a)

    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#C9C3B6"), name="Perfect")
    for n, colr in zip(MODELS, [FOREST, SAND, GREY]):
        pt, pp = calibration_curve(y_te, probs[n], n_bins=8, strategy="quantile")
        fig.add_scatter(x=pp, y=pt, mode="lines+markers", name=n, line=dict(color=colr, width=2.2 if n == model_sel else 1.3),
                        marker=dict(size=7 if n == model_sel else 5))
    fig.update_layout(title="Calibration (reliability) curve", xaxis_title="Mean predicted probability",
                      yaxis_title="Observed frequency", legend=dict(x=.02, y=.98, font=dict(size=10)))
    show(fig, 380, b)

    a, b = st.columns([3, 2])
    sub = imp[imp["model"] == model_sel].sort_values("importance_mean")
    fig = go.Figure(go.Bar(x=sub["importance_mean"], y=[lab(f) for f in sub["feature"]], orientation="h",
                           error_x=dict(type="data", array=sub["importance_std"], color=GREY, thickness=1),
                           marker=dict(color=sub["importance_mean"], colorscale=SEQ, line=dict(width=0))))
    fig.update_layout(title=f"Permutation importance · {model_sel}", xaxis_title="Drop in ROC-AUC")
    show(fig, 500, a)
    rows = pd.DataFrame([{"Model": n, "CV AUC": f"{m['cv_roc_auc_mean']:.3f} ± {m['cv_roc_auc_sd']:.3f}", "Test AUC": m["test_roc_auc"],
                          "Accuracy": m["test_accuracy"], "Precision": m["test_precision"], "Recall": m["test_recall"], "F1": m["test_f1"]}
                         for n, m in metrics["models"].items()])
    b.markdown('<div class="np-sec">Model comparison · threshold 0.50</div>', unsafe_allow_html=True)
    b.dataframe(rows, hide_index=True, use_container_width=True)
    abl = pd.DataFrame([{"Feature set": k_, "CV AUC": v["cv_roc_auc_mean"], "sd": v["cv_roc_auc_sd"]}
                        for k_, v in metrics["ablation_logistic"].items()])
    fig = px.bar(abl, x="Feature set", y="CV AUC", error_y="sd", text="CV AUC", color="CV AUC", color_continuous_scale=SEQ)
    fig.update_layout(title="Feature-set ablation", coloraxis_showscale=False, yaxis_range=[.78, .87], xaxis_title=None)
    show(fig, 320, b)


# ============================================================================
# DATA
# ============================================================================
elif page == "Data":
    hero("Dataset", "NHANES 2011-2014 Analytic Cohort", ["CDC / NCHS", "Public domain", f"{len(df):,} participants", f"{len(NUMERIC)} features"])
    c = st.columns(4)
    kpi(c[0], "Participants", f"{len(df):,}", "aged 60-80")
    kpi(c[1], "Features", f"{len(NUMERIC)}", "analysis variables")
    kpi(c[2], "Complete rows", f"{df[NUMERIC].notna().all(axis=1).mean() * 100:.1f}%", "no missing value")
    kpi(c[3], "Low processing speed", f"{df[TARGET].mean() * 100:.1f}%", f"DSST < {CUTOFF:.0f}")
    st.write("")

    # Radio instead of st.tabs: dataframes inside tabs can trigger a resize loop (React error #185).
    view_sel = st.radio("View", ["Table", "Statistics", "Missing values"], horizontal=True, label_visibility="collapsed")
    if view_sel == "Table":
        f1, f2, f3 = st.columns([2, 1, 1])
        groups_sel = f1.multiselect("Feature groups", list(GROUPS), default=["Cognitive", "Motor", "Demographics"])
        sex_sel = f2.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])
        age_rng = f3.slider("Age", 60, 80, (60, 80))
        cols = ["participant_id"] + [c_ for g in groups_sel for c_ in GROUPS[g]]
        view = df.loc[dfp["Sex"].isin(sex_sel) & dfp["age"].between(*age_rng), list(dict.fromkeys(cols))]
        st.caption(f"{len(view):,} rows")
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", view.to_csv(index=False), "neuroperform_data.csv", "text/csv")
    elif view_sel == "Statistics":
        g = st.selectbox("Feature group", list(GROUPS))
        desc = df[GROUPS[g]].describe().T
        desc.insert(0, "variable", [lab(i) for i in desc.index])
        st.dataframe(desc.round(2), use_container_width=True, hide_index=True)
        by = st.radio("Group by", ["Sex", "Age group", "Processing speed"], horizontal=True)
        st.dataframe(dfp.groupby(by)[GROUPS[g]].mean().T.rename(index=lab).round(2), use_container_width=True)
    else:
        m = (df[NUMERIC].isna().mean() * 100)
        m = m[m > 0].sort_values()
        fig = go.Figure(go.Bar(x=m.values, y=[lab(i) for i in m.index], orientation="h",
                               marker=dict(color=m.values, colorscale=[[0, BEIGE], [1, CLAY]]),
                               text=[f"{v:.2f}%" for v in m.values], textposition="outside"))
        fig.update_layout(title="Missing values (imputed with training median inside each model)", xaxis_title="Missing (%)")
        show(fig, 360)
