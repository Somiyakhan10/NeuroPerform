"""NeuroPerform pipeline: download, clean, engineer features, statistics, ML, save.  Run: python src/pipeline.py"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # silences a joblib warning on Windows

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

CLEAN_CSV = PROCESSED_DIR / "neuroperform_clean.csv"
MODEL_FILE = MODEL_DIR / "neuroperform_model.joblib"
METRICS_FILE = RESULTS_DIR / "model_metrics.json"
STATS_FILE = RESULTS_DIR / "statistical_results.json"
IMPORTANCE_FILE = RESULTS_DIR / "feature_importance.csv"

RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# Data source: NHANES (CDC National Center for Health Statistics)
# ----------------------------------------------------------------------------
# Two survey cycles in which the cognitive battery (adults 60+) and the
# hand-grip dynamometry exam were both administered.
CYCLES = {"G": "2011", "H": "2013"}  # suffix -> start year used in the URL
COMPONENTS = {
    "DEMO": "Demographics",
    "CFQ": "Cognitive Functioning (CERAD, Animal Fluency, DSST)",
    "MGX": "Muscle Strength - Grip Test",
    "PAQ": "Physical Activity questionnaire",
    "DPQ": "Mental Health - Depression Screener (PHQ-9)",
    "SLQ": "Sleep Disorders questionnaire",
    "BMX": "Body Measures",
}
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}_{suffix}.xpt"


def download_raw_data(raw_dir: Path = RAW_DIR, verbose: bool = True) -> None:
    """Download every NHANES .xpt file that is not already on disk."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for suffix, year in CYCLES.items():
        for name in COMPONENTS:
            dest = raw_dir / f"{name}_{suffix}.xpt"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            url = BASE_URL.format(year=year, name=name, suffix=suffix)
            if verbose:
                print(f"Downloading {url}")
            urllib.request.urlretrieve(url, dest)


def _read_xpt(path: Path) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport")
    # SAS transport files store exact zero as ~5.4e-79; round those back to 0.
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-10, 0.0)
    return df


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read all components for both cycles and merge them on SEQN (participant ID)."""
    cycles = []
    for suffix in CYCLES:
        merged = None
        for name in COMPONENTS:
            df = _read_xpt(raw_dir / f"{name}_{suffix}.xpt")
            merged = df if merged is None else merged.merge(df, on="SEQN", how="left", suffixes=("", f"_{name}"))
        merged["cycle"] = f"{CYCLES[suffix]}-{int(CYCLES[suffix]) + 1}"
        cycles.append(merged)
    return pd.concat(cycles, ignore_index=True, sort=False)


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def engineer_grip_features(hand1: np.ndarray, hand2: np.ndarray) -> dict:
    """
    Derive motor-function features from three maximal grip trials per hand (kg).

    hand1, hand2 : arrays of shape (n, 3) - trials 1, 2, 3 for each hand.

    Returns a dict of 1-D arrays:
      grip_combined_kg     best trial hand 1 + best trial hand 2 (NHANES MGDCGSZ definition)
      grip_max_kg          single strongest trial
      grip_asymmetry_pct   |best1 - best2| / max(best1, best2) * 100
      grip_cv_pct          trial-to-trial coefficient of variation (mean of both hands)
      grip_change_pct      (trial 3 - trial 1) / trial 1 * 100, averaged over hands
                           -> negative = force decrement across repeated efforts
    """
    hand1 = np.asarray(hand1, dtype=float).reshape(-1, 3)
    hand2 = np.asarray(hand2, dtype=float).reshape(-1, 3)
    best1, best2 = hand1.max(axis=1), hand2.max(axis=1)
    cv1 = hand1.std(axis=1, ddof=1) / hand1.mean(axis=1) * 100
    cv2 = hand2.std(axis=1, ddof=1) / hand2.mean(axis=1) * 100
    ch1 = (hand1[:, 2] - hand1[:, 0]) / hand1[:, 0] * 100
    ch2 = (hand2[:, 2] - hand2[:, 0]) / hand2[:, 0] * 100
    return {
        "grip_combined_kg": best1 + best2,
        "grip_max_kg": np.maximum(best1, best2),
        "grip_asymmetry_pct": np.abs(best1 - best2) / np.maximum(best1, best2) * 100,
        "grip_cv_pct": (cv1 + cv2) / 2,
        "grip_change_pct": (ch1 + ch2) / 2,
    }


def _clean_codes(s: pd.Series, invalid: tuple) -> pd.Series:
    """Replace NHANES 'refused' / 'don't know' codes with NaN."""
    return s.where(~s.isin(invalid))


def build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply inclusion criteria, clean codes and engineer features."""
    flow = {"merged_participants": int(len(raw))}

    df = raw[raw["RIDAGEYR"] >= 60].copy()
    flow["age_60_plus"] = int(len(df))
    df = df[df["CFDDS"].notna()]
    flow["with_dsst_score"] = int(len(df))

    trial_cols_h1 = ["MGXH1T1", "MGXH1T2", "MGXH1T3"]
    trial_cols_h2 = ["MGXH2T1", "MGXH2T2", "MGXH2T3"]
    df = df.dropna(subset=trial_cols_h1 + trial_cols_h2)
    flow["with_all_6_grip_trials"] = int(len(df))

    out = pd.DataFrame(index=df.index)
    out["participant_id"] = df["SEQN"].astype(int)
    out["cycle"] = df["cycle"]

    # --- demographics / anthropometrics ---
    out["age"] = df["RIDAGEYR"]  # NHANES top-codes age at 80
    out["sex_female"] = (df["RIAGENDR"] == 2).astype(int)
    out["education"] = _clean_codes(df["DMDEDUC2"], (7, 9))  # 1 = <9th grade ... 5 = college graduate
    out["bmi"] = df["BMXBMI"]

    # --- cognitive outcomes ---
    out["dsst_score"] = df["CFDDS"]  # Digit Symbol Substitution: correct symbols in 120 s
    out["animal_fluency"] = df["CFDAST"]  # animals named in 60 s
    out["cerad_immediate"] = df[["CFDCST1", "CFDCST2", "CFDCST3"]].sum(axis=1, min_count=3)
    out["cerad_delayed"] = df["CFDCSR"]

    # --- motor function: grip-strength trials + engineered features ---
    for c in trial_cols_h1 + trial_cols_h2:
        out[c.lower()] = df[c]
    grip = engineer_grip_features(df[trial_cols_h1].values, df[trial_cols_h2].values)
    for k, v in grip.items():
        out[k] = v
    out["grip_per_bmi"] = out["grip_combined_kg"] / out["bmi"]

    # --- fatigue / sleep / mood ---
    out["tiredness_score"] = _clean_codes(df["DPQ040"], (7, 9))  # PHQ-9 item 4: 0 = not at all ... 3 = nearly every day
    other_items = [f"DPQ0{i}0" for i in (1, 2, 3, 5, 6, 7, 8, 9)]
    phq = df[other_items].apply(lambda s: _clean_codes(s, (7, 9)))
    out["depressive_symptoms"] = phq.sum(axis=1, min_count=len(other_items))  # PHQ-9 minus the tiredness item (0-24)
    out["sleep_hours"] = _clean_codes(df["SLD010H"], (77, 99))
    out["sleep_trouble"] = _clean_codes(df["SLQ050"], (7, 9)).map({1: 1, 2: 0})

    # --- recreational physical activity ---
    vig = _clean_codes(df["PAQ650"], (7, 9)).map({1: 1, 2: 0})
    mod = _clean_codes(df["PAQ665"], (7, 9)).map({1: 1, 2: 0})
    vig_min = _clean_codes(df["PAQ655"], (77, 99)) * _clean_codes(df["PAD660"], (7777, 9999))
    mod_min = _clean_codes(df["PAQ670"], (77, 99)) * _clean_codes(df["PAD675"], (7777, 9999))
    vig_min = vig_min.where(vig != 0, 0.0)
    mod_min = mod_min.where(mod != 0, 0.0)
    out["vigorous_recreation"] = vig
    out["moderate_recreation"] = mod
    out["recreation_min_week"] = (vig_min + mod_min).clip(upper=3000)  # cap implausible extremes

    # --- outcome label ---
    cutoff = float(out["dsst_score"].quantile(0.25))
    out["low_processing_speed"] = (out["dsst_score"] < cutoff).astype(int)
    flow["final_analytic_sample"] = int(len(out))
    flow["dsst_low_cutoff"] = cutoff

    return out.reset_index(drop=True), flow


# Features used by the machine-learning models (cognitive scores are excluded on purpose:
# the question is whether motor, fatigue, activity and demographic factors predict cognition).
MODEL_FEATURES = [
    "age", "sex_female", "education", "bmi",
    "grip_combined_kg", "grip_asymmetry_pct", "grip_cv_pct", "grip_change_pct", "grip_per_bmi",
    "tiredness_score", "depressive_symptoms", "sleep_hours", "sleep_trouble",
    "vigorous_recreation", "moderate_recreation", "recreation_min_week",
]
DEMOGRAPHIC_FEATURES = ["age", "sex_female", "education", "bmi"]
TARGET = "low_processing_speed"

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex_female": "Sex (female = 1)",
    "education": "Education level (1-5)",
    "bmi": "Body-mass index",
    "dsst_score": "DSST processing-speed score",
    "animal_fluency": "Animal fluency (words/60 s)",
    "cerad_immediate": "CERAD immediate recall (0-30)",
    "cerad_delayed": "CERAD delayed recall (0-10)",
    "grip_combined_kg": "Combined grip strength (kg)",
    "grip_max_kg": "Maximal grip strength (kg)",
    "grip_asymmetry_pct": "Grip asymmetry (%)",
    "grip_cv_pct": "Grip trial variability, CV (%)",
    "grip_change_pct": "Grip change trial 1 to 3 (%)",
    "grip_per_bmi": "Grip strength / BMI",
    "tiredness_score": "Self-reported tiredness (0-3)",
    "depressive_symptoms": "Depressive symptoms (PHQ-8, 0-24)",
    "sleep_hours": "Usual sleep (h/night)",
    "sleep_trouble": "Reported trouble sleeping",
    "vigorous_recreation": "Vigorous recreational activity",
    "moderate_recreation": "Moderate recreational activity",
    "recreation_min_week": "Recreational activity (min/week)",
    "low_processing_speed": "Low processing speed (label)",
}


# ----------------------------------------------------------------------------
# Statistical analysis
# ----------------------------------------------------------------------------
def run_statistics(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    res: dict = {}

    # 1. Spearman correlations between cognitive, motor, fatigue variables
    corr_vars = ["dsst_score", "animal_fluency", "cerad_immediate", "cerad_delayed",
                 "grip_combined_kg", "grip_cv_pct", "grip_asymmetry_pct", "grip_change_pct",
                 "tiredness_score", "sleep_hours", "recreation_min_week", "age", "education"]
    rho = df[corr_vars].corr(method="spearman")
    res["spearman_matrix"] = rho.round(3).to_dict()
    key_pairs = [("grip_combined_kg", "dsst_score"), ("grip_cv_pct", "dsst_score"),
                 ("tiredness_score", "dsst_score"), ("recreation_min_week", "dsst_score"),
                 ("age", "dsst_score"), ("grip_combined_kg", "cerad_delayed"),
                 ("grip_combined_kg", "animal_fluency")]
    pairs = []
    for a, b in key_pairs:
        sub = df[[a, b]].dropna()
        r, p = stats.spearmanr(sub[a], sub[b])
        pairs.append({"x": a, "y": b, "rho": round(float(r), 3), "p": float(p), "n": int(len(sub))})
    res["key_correlations"] = pairs

    # 2. Adjusted association: DSST ~ grip (z) + age + sex + education + BMI  (OLS)
    d = df[["dsst_score", "grip_combined_kg", "age", "sex_female", "education", "bmi",
            "tiredness_score", "recreation_min_week"]].dropna().copy()
    d["grip_z"] = (d["grip_combined_kg"] - d["grip_combined_kg"].mean()) / d["grip_combined_kg"].std()
    m1 = smf.ols("dsst_score ~ grip_z + age + sex_female + C(education) + bmi", data=d).fit()
    m2 = smf.ols("dsst_score ~ grip_z + age + sex_female + C(education) + bmi + tiredness_score"
                 " + np.log1p(recreation_min_week)", data=d).fit()

    def _coef(model, name):
        ci = model.conf_int().loc[name]
        return {"beta": round(float(model.params[name]), 3), "ci_low": round(float(ci[0]), 3),
                "ci_high": round(float(ci[1]), 3), "p": float(model.pvalues[name])}

    res["ols_grip_adjusted"] = {
        "n": int(m1.nobs), "r2": round(float(m1.rsquared), 3),
        "grip_per_sd": _coef(m1, "grip_z"), "age_per_year": _coef(m1, "age"),
    }
    res["ols_full"] = {
        "n": int(m2.nobs), "r2": round(float(m2.rsquared), 3),
        "grip_per_sd": _coef(m2, "grip_z"), "tiredness_per_point": _coef(m2, "tiredness_score"),
        "log_recreation": _coef(m2, "np.log1p(recreation_min_week)"),
    }

    # 3. DSST by vigorous recreational activity (Mann-Whitney U, rank-biserial r)
    a = df.loc[df["vigorous_recreation"] == 1, "dsst_score"].dropna()
    b = df.loc[df["vigorous_recreation"] == 0, "dsst_score"].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    res["dsst_by_vigorous_activity"] = {
        "median_active": float(a.median()), "median_inactive": float(b.median()),
        "n_active": int(len(a)), "n_inactive": int(len(b)),
        "U": float(u), "p": float(p), "rank_biserial": round(float(2 * u / (len(a) * len(b)) - 1), 3),  # >0: active group higher
    }

    # 4. DSST and grip across self-reported tiredness levels (Kruskal-Wallis, epsilon^2)
    for var in ["dsst_score", "grip_combined_kg"]:
        groups = [g[var].dropna() for _, g in df.groupby("tiredness_score")]
        h, p = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        res[f"{var}_by_tiredness"] = {
            "H": round(float(h), 3), "p": float(p), "epsilon_sq": round(float(h / (n - 1)), 4),
            "medians": {str(int(k)): float(g[var].median()) for k, g in df.groupby("tiredness_score")},
            "n_per_level": {str(int(k)): int(g[var].notna().sum()) for k, g in df.groupby("tiredness_score")},
        }

    # 5. Repeated-effort motor pattern: grip trial 1 vs 2 vs 3 (Friedman test, per hand)
    rep = {}
    for hand in ("h1", "h2"):
        cols = [f"mgx{hand}t{i}" for i in (1, 2, 3)]
        sub = df[cols].dropna()
        chi, p = stats.friedmanchisquare(*[sub[c] for c in cols])
        rep[hand] = {"means": [round(float(sub[c].mean()), 2) for c in cols], "chi2": round(float(chi), 2),
                     "p": float(p), "pct_with_decrement": round(float((sub[cols[2]] < sub[cols[0]]).mean() * 100), 1)}
    res["grip_repeated_trials"] = rep

    # 6. Variability: coefficient of variation of DSST and grip by age group
    bins = pd.cut(df["age"], [59, 64, 69, 74, 80], labels=["60-64", "65-69", "70-74", "75-80"])
    var_tbl = df.groupby(bins, observed=True).agg(
        n=("dsst_score", "size"), dsst_mean=("dsst_score", "mean"), dsst_sd=("dsst_score", "std"),
        grip_mean=("grip_combined_kg", "mean"), grip_sd=("grip_combined_kg", "std"),
        grip_trial_cv=("grip_cv_pct", "mean"))
    var_tbl["dsst_cv_pct"] = var_tbl["dsst_sd"] / var_tbl["dsst_mean"] * 100
    var_tbl["grip_cv_between_pct"] = var_tbl["grip_sd"] / var_tbl["grip_mean"] * 100
    res["variability_by_age"] = var_tbl.round(2).reset_index().rename(columns={"age": "age_group"}).to_dict(orient="records")
    r, p = stats.spearmanr(df["grip_cv_pct"], df["age"])
    res["grip_cv_vs_age"] = {"rho": round(float(r), 3), "p": float(p)}
    return res


# ----------------------------------------------------------------------------
# Machine learning
# ----------------------------------------------------------------------------
def make_models() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10,
                                             class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                     l2_regularization=1.0, class_weight="balanced",
                                                     random_state=RANDOM_STATE)),
        ]),
    }


def train_and_evaluate(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score, roc_curve)
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

    X, y = df[MODEL_FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    metrics = {"n_train": int(len(X_tr)), "n_test": int(len(X_te)),
               "positive_rate": round(float(y.mean()), 3), "features": MODEL_FEATURES, "models": {}}
    fitted = {}
    for name, model in make_models().items():
        cvr = cross_validate(clone(model), X_tr, y_tr, cv=cv, scoring=["roc_auc", "f1", "balanced_accuracy"])
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_te, proba)
        metrics["models"][name] = {
            "cv_roc_auc_mean": round(float(cvr["test_roc_auc"].mean()), 3),
            "cv_roc_auc_sd": round(float(cvr["test_roc_auc"].std()), 3),
            "cv_f1_mean": round(float(cvr["test_f1"].mean()), 3),
            "cv_balanced_acc_mean": round(float(cvr["test_balanced_accuracy"].mean()), 3),
            "test_accuracy": round(float(accuracy_score(y_te, pred)), 3),
            "test_precision": round(float(precision_score(y_te, pred)), 3),
            "test_recall": round(float(recall_score(y_te, pred)), 3),
            "test_f1": round(float(f1_score(y_te, pred)), 3),
            "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 3),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
            "roc_curve": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
        }
        fitted[name] = model

    # Model selection uses cross-validated AUC on the training set (not the test set).
    best_name = max(metrics["models"], key=lambda k: metrics["models"][k]["cv_roc_auc_mean"])
    metrics["best_model"] = best_name

    # Ablation: how much do motor + fatigue + activity features add beyond demographics?
    abl = {}
    for label, feats in [("Demographics only", DEMOGRAPHIC_FEATURES),
                         ("Demographics + motor", DEMOGRAPHIC_FEATURES + [
                             "grip_combined_kg", "grip_asymmetry_pct", "grip_cv_pct", "grip_change_pct", "grip_per_bmi"]),
                         ("All features", MODEL_FEATURES)]:
        m = clone(make_models()["Logistic Regression"])
        s = cross_validate(m, X_tr[feats], y_tr, cv=cv, scoring="roc_auc")["test_score"]
        abl[label] = {"cv_roc_auc_mean": round(float(s.mean()), 3), "cv_roc_auc_sd": round(float(s.std()), 3)}
    metrics["ablation_logistic"] = abl

    # Feature importance: permutation importance (AUC drop) on the held-out test set for every model
    rows = []
    for name, model in fitted.items():
        pi = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=20,
                                    random_state=RANDOM_STATE, n_jobs=-1)
        for f, mu, sd in zip(MODEL_FEATURES, pi.importances_mean, pi.importances_std):
            rows.append({"model": name, "feature": f, "importance_mean": float(mu), "importance_std": float(sd)})
    imp = pd.DataFrame(rows)

    # Logistic-regression standardized coefficients (direction of effect)
    lr = fitted["Logistic Regression"].named_steps["model"]
    metrics["logistic_coefficients"] = {f: round(float(c), 3) for f, c in zip(MODEL_FEATURES, lr.coef_[0])}

    # Training-set ranges used to validate inputs on the prediction page
    metrics["feature_ranges"] = {
        f: {"min": float(X[f].min()), "max": float(X[f].max()), "median": float(X[f].median())} for f in MODEL_FEATURES
    }
    return fitted, metrics, imp


# ----------------------------------------------------------------------------
# Static figures (for the README / report; the dashboard draws its own Plotly charts)
# ----------------------------------------------------------------------------
def save_figures(df: pd.DataFrame, metrics: dict, imp: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sage, beige, grey, ink = "#7C9A7E", "#C8B79A", "#9AA0A6", "#2F3A33"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["dsst_score"], bins=35, color=sage, edgecolor="white")
    ax.axvline(metrics.get("dsst_cutoff", df["dsst_score"].quantile(0.25)), color=ink, ls="--", lw=1)
    ax.set(xlabel="DSST score (correct symbols / 120 s)", ylabel="Participants", title="Processing-speed distribution")
    fig.tight_layout(); fig.savefig(FIG_DIR / "dsst_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for sex, c, lab in [(0, beige, "Male"), (1, sage, "Female")]:
        s = df[df["sex_female"] == sex]
        ax.scatter(s["grip_combined_kg"], s["dsst_score"], s=8, alpha=0.45, color=c, label=lab)
    ax.set(xlabel="Combined grip strength (kg)", ylabel="DSST score", title="Grip strength vs processing speed")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "grip_vs_dsst.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, c in zip(metrics["models"], [grey, beige, sage]):
        rc = metrics["models"][name]["roc_curve"]
        ax.plot(rc["fpr"], rc["tpr"], color=c, lw=2, label=f"{name} (AUC {metrics['models'][name]['test_roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], color="#ccc", ls=":")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves (held-out test set)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "roc_curves.png", dpi=150); plt.close(fig)

    best = metrics["best_model"]
    sub = imp[imp["model"] == best].sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([FEATURE_LABELS[f] for f in sub["feature"]], sub["importance_mean"], xerr=sub["importance_std"],
            color=sage, ecolor=grey)
    ax.set(xlabel="Mean drop in ROC-AUC when permuted", title=f"Permutation importance - {best}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    cm = np.array(metrics["models"][best]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Greens")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color=ink, fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Typical", "Low"], yticklabels=["Typical", "Low"],
           xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix - {best}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def _to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run_pipeline(verbose: bool = True) -> dict:
    download_raw_data(verbose=verbose)
    raw = load_raw()
    df, flow = build_dataset(raw)
    for d in (PROCESSED_DIR, MODEL_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)

    st = run_statistics(df)
    st["participant_flow"] = flow
    models, metrics, imp = train_and_evaluate(df)
    metrics["dsst_cutoff"] = flow["dsst_low_cutoff"]

    # One reusable file: the best model plus the other two fitted pipelines for comparison.
    joblib.dump({"model": models[metrics["best_model"]], "model_name": metrics["best_model"],
                 "all_models": models, "features": MODEL_FEATURES,
                 "dsst_cutoff": flow["dsst_low_cutoff"]}, MODEL_FILE, compress=3)
    STATS_FILE.write_text(json.dumps(_to_jsonable(st), indent=2))
    METRICS_FILE.write_text(json.dumps(_to_jsonable(metrics), indent=2))
    imp.to_csv(IMPORTANCE_FILE, index=False)
    save_figures(df, metrics, imp)

    if verbose:
        print("\nParticipant flow:", json.dumps(flow, indent=2))
        print("\nModel comparison:")
        for k, v in metrics["models"].items():
            print(f"  {k:20s} CV AUC {v['cv_roc_auc_mean']:.3f} +/- {v['cv_roc_auc_sd']:.3f} | "
                  f"test AUC {v['test_roc_auc']:.3f} | F1 {v['test_f1']:.3f} | recall {v['test_recall']:.3f}")
        print("Best model (by CV AUC):", metrics["best_model"])
        print("Ablation:", metrics["ablation_logistic"])
        print(f"\nSaved: {CLEAN_CSV.name}, {MODEL_FILE.name}, {METRICS_FILE.name}, {STATS_FILE.name}, figures/")
    return {"flow": flow, "stats": st, "metrics": metrics}


if __name__ == "__main__":
    run_pipeline()
