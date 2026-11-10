from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

# ============================================================
# ChurnIQ | Customer Retention Intelligence
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_GROUPS = {
    "Demographics": ["gender", "SeniorCitizen", "Partner", "Dependents"],
    "Subscription": ["tenure", "Contract", "PhoneService", "MultipleLines", "InternetService"],
    "Add-on services": [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ],
    "Billing": ["PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"],
}

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

INTERNET_FEATURES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

TIER_COLORS = {"Low": "#48d6a0", "Medium": "#fbbf24", "High": "#fb7185"}
PAGES = [
    "Overview", "Single Customer", "Batch Scoring",
    "Model Performance", "Activity", "Model Information",
]
PAGE_ICONS = {
    "Overview": "📊", "Single Customer": "👤", "Batch Scoring": "📁",
    "Model Performance": "📈", "Activity": "🕘", "Model Information": "🧠",
}

# ============================================================
# Theme
# ============================================================

st.markdown(
    """
<style>
:root {
    --bg: #0a0f1d; --surface: #121a2e; --surface-2: #18223a;
    --border: #28365a; --text: #f3f5ff; --muted: #9eabc7;
    --purple: #9b7bfa; --blue: #54c7ec; --green: #48d6a0;
    --amber: #fbbf24; --red: #fb7185;
}
.stApp {
    background:
        radial-gradient(ellipse at 85% -5%, rgba(115,78,210,.20), transparent 40%),
        radial-gradient(ellipse at 0% 75%, rgba(40,106,160,.13), transparent 40%),
        var(--bg);
    color: var(--text);
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1380px; padding-top: 1.5rem; padding-bottom: 3rem; }
h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -0.4px; }
[data-testid="stSidebar"] { background: #0d1426; border-right: 1px solid var(--border); }

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #18223a, #121a2e);
    border: 1px solid var(--border); border-radius: 16px; padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--text); font-weight: 800; }

button[data-baseweb="tab"] { font-weight: 600; }

div.stButton > button[kind="primary"],
div.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(100deg, #7555ed, #a855f7);
    color: white; border: 0; border-radius: 11px; min-height: 46px; font-weight: 700;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(100deg, #8b6df5, #bb70f7); color: white;
}

.hero {
    background:
        radial-gradient(circle at 92% 10%, rgba(167,139,250,.25), transparent 40%),
        linear-gradient(115deg, #252253, #172344 68%, #11182e);
    border: 1px solid #454477; border-radius: 22px;
    padding: 34px 36px; margin-bottom: 24px;
}
.hero-label { color: #c8b9ff; font-size: 12px; font-weight: 700; letter-spacing: .6px; }
.hero-title {
    color: white; font-size: clamp(30px, 4vw, 46px); font-weight: 850;
    letter-spacing: -1.6px; margin: 10px 0 8px;
}
.hero-copy { color: #c4cee8; font-size: 15px; line-height: 1.7; max-width: 760px; }
.badges { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }
.badge {
    background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14);
    color: #dfe5fb; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 600;
}

.kpi {
    background: linear-gradient(145deg, #18223a, #121a2e);
    border: 1px solid var(--border); border-left: 3px solid var(--accent, #9b7bfa);
    border-radius: 14px; padding: 16px 20px; height: 100%;
}
.kpi-label { color: var(--muted); font-size: 13px; font-weight: 600; }
.kpi-value { color: white; font-size: 30px; font-weight: 800; letter-spacing: -.8px; margin: 4px 0 2px; }
.kpi-sub { color: #7f8ba7; font-size: 12px; }

.panel {
    background: linear-gradient(145deg, #18223a, #121a2e);
    border: 1px solid var(--border); border-radius: 16px; padding: 20px;
}
.step-title { color: white; font-weight: 750; font-size: 16px; margin-bottom: 4px; }
.step-copy { color: var(--muted); font-size: 14px; line-height: 1.6; }

.tier-pill {
    display: inline-block; padding: 5px 16px; border-radius: 999px;
    font-weight: 800; font-size: 13px; border: 1px solid;
}
.signal {
    background: #141d33; border: 1px solid var(--border);
    border-radius: 12px; padding: 12px 16px; margin-bottom: 10px;
}
.signal b { color: white; }
.signal span { color: var(--muted); font-size: 14px; }
.muted { color: var(--muted); }
.footer { color: #7f8ba7; text-align: center; font-size: 12px; padding: 28px 0 5px; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# UI helpers
# ============================================================


def hero(label: str, title: str, copy: str, badges: list[str] | None = None) -> None:
    badge_html = "".join(f'<span class="badge">{b}</span>' for b in (badges or []))
    st.markdown(
        f'<div class="hero"><div class="hero-label">{label}</div>'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-copy">{copy}</div>'
        f'<div class="badges">{badge_html}</div></div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = "", accent: str = "#9b7bfa") -> None:
    st.markdown(
        f'<div class="kpi" style="--accent:{accent}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def step_card(title: str, copy: str) -> None:
    st.markdown(
        f'<div class="panel"><div class="step-title">{title}</div>'
        f'<div class="step-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c4cee8"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "y": -0.15},
    )
    fig.update_xaxes(gridcolor="#223052", zeroline=False)
    fig.update_yaxes(gridcolor="#223052", zeroline=False)
    return fig


def gauge(probability: float, threshold: float) -> go.Figure:
    medium_start = threshold * 0.6 * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 46, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#51608a"},
                "bar": {"color": TIER_COLORS[risk_tier(probability, threshold)], "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, medium_start], "color": "rgba(72,214,160,.18)"},
                    {"range": [medium_start, threshold * 100], "color": "rgba(251,191,36,.18)"},
                    {"range": [threshold * 100, 100], "color": "rgba(251,113,133,.18)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        )
    )
    return style_fig(fig, 270)


def tier_pill(tier: str) -> str:
    color = TIER_COLORS[tier]
    return (
        f'<span class="tier-pill" style="color:{color};border-color:{color};'
        f'background:{color}1f">{tier} risk</span>'
    )


# ============================================================
# Model helpers
# ============================================================


@st.cache_resource
def load_assets():
    model_path = BASE_DIR / "churn_model.pkl"
    threshold_path = BASE_DIR / "churn_threshold.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.name}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path.name}")

    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path))

    if not 0 <= threshold <= 1:
        raise ValueError("The saved decision threshold must be between 0 and 1.")
    return model, threshold


def get_positive_class_index(model) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("The loaded model does not expose classes_.")

    normalized = [str(value).strip().lower() for value in classes]
    for candidate in ("yes", "true", "1", "churn"):
        if candidate in normalized:
            return normalized.index(candidate)
    if len(classes) == 2:
        return 1
    raise ValueError(f"Could not identify the positive churn class in {list(classes)}.")


def score_customers(model, customers: pd.DataFrame) -> list[float]:
    missing = [c for c in FEATURE_COLUMNS if c not in customers]
    if missing:
        raise ValueError(f"Missing required model columns: {', '.join(missing)}")

    probabilities = model.predict_proba(customers[FEATURE_COLUMNS].copy())
    return probabilities[:, get_positive_class_index(model)].astype(float).tolist()


def build_customer_row(values: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{column: values[column] for column in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS
    )


def risk_tier(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= threshold * 0.6:
        return "Medium"
    return "Low"


def prepare_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning so common CSV quirks don't break scoring."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "TotalCharges" in df:
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)  # blank for tenure = 0

    if "SeniorCitizen" in df and df["SeniorCitizen"].dtype == object:
        df["SeniorCitizen"] = (
            df["SeniorCitizen"].astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0})
        )
    return df


def sample_data() -> pd.DataFrame:
    base = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5, "TotalCharges": 179.0,
    }
    loyal = {
        **base, "gender": "Male", "tenure": 60, "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
        "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Two year", "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 64.9, "TotalCharges": 3894.0,
    }
    middle = {
        **base, "tenure": 18, "Contract": "One year", "InternetService": "DSL",
        "TechSupport": "Yes", "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 58.2, "TotalCharges": 1047.6,
    }
    return pd.DataFrame([base, loyal, middle], columns=FEATURE_COLUMNS)


def risk_signals(v: dict) -> list[tuple[str, str]]:
    """Rule-based retention signals. These are business heuristics, not model explanations."""
    signals: list[tuple[str, str]] = []
    if v["Contract"] == "Month-to-month":
        signals.append(("Month-to-month contract", "Offer a discounted one-year plan."))
    if v["tenure"] < 12:
        signals.append(("Early-tenure customer (under 12 months)", "Schedule an onboarding check-in."))
    if v["InternetService"] == "Fiber optic" and v["TechSupport"] == "No":
        signals.append(("Fiber customer without tech support", "Offer a trial of tech support."))
    if v["InternetService"] != "No" and v["OnlineSecurity"] == "No":
        signals.append(("No online security add-on", "Bundle security into the current plan."))
    if v["PaymentMethod"] == "Electronic check":
        signals.append(("Pays by electronic check", "Incentivize automatic payment."))
    if v["MonthlyCharges"] > 80:
        signals.append(("High monthly charges", "Review plan fit and price sensitivity."))
    return signals


def run_scenarios(model, values: dict, base_probability: float) -> pd.DataFrame:
    rows = []

    def add(name: str, changes: dict) -> None:
        p = score_customers(model, build_customer_row({**values, **changes}))[0]
        rows.append({
            "Scenario": name,
            "Churn probability (%)": p * 100,
            "Change (pp)": (p - base_probability) * 100,
        })

    if values["Contract"] == "Month-to-month":
        add("Move to a one-year contract", {"Contract": "One year"})
    if values["Contract"] != "Two year":
        add("Move to a two-year contract", {"Contract": "Two year"})
    if values["PaymentMethod"] in ("Electronic check", "Mailed check"):
        add("Switch to automatic bank transfer", {"PaymentMethod": "Bank transfer (automatic)"})
    if values["InternetService"] != "No" and values["TechSupport"] == "No":
        add("Add tech support", {"TechSupport": "Yes"})
    if values["InternetService"] != "No" and values["OnlineSecurity"] == "No":
        add("Add online security", {"OnlineSecurity": "Yes"})
    return pd.DataFrame(rows)



def _original_feature(encoded_name: str) -> tuple[str, str]:
    """Map 'cat__Contract_Month-to-month' -> ('Contract', 'Month-to-month')."""
    name = encoded_name.split("__", 1)[-1]
    for col in sorted(FEATURE_COLUMNS, key=len, reverse=True):
        if name == col:
            return col, ""
        if name.startswith(col + "_"):
            return col, name[len(col) + 1:]
    return name, ""


def linear_parts(model):
    """Return (preprocessor, feature_names, coefficients) for preprocessor + linear classifier."""
    try:
        pre = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]
        sign = 1 if get_positive_class_index(model) == 1 else -1
        return pre, list(pre.get_feature_names_out()), np.asarray(clf.coef_[0]) * sign
    except Exception:
        return None


def explain_customer(model, customer_df: pd.DataFrame):
    """Exact per-feature impact on churn log-odds for a linear model.

    Numeric features are measured against the training average (StandardScaler),
    categorical features against the average category of that feature.
    """
    parts = linear_parts(model)
    if parts is None:
        return None
    pre, names, coefs = parts

    X = pre.transform(customer_df[FEATURE_COLUMNS])
    x = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)[0]

    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        groups.setdefault(_original_feature(name)[0], []).append(i)

    rows = []
    for feature, idx in groups.items():
        impact = float(np.sum(coefs[idx] * x[idx]))
        if len(idx) > 1:  # one-hot group: compare with the average category
            impact -= float(np.mean(coefs[idx]))
        value = customer_df[feature].iloc[0] if feature in customer_df else ""
        if isinstance(value, float):
            value = f"{value:,.1f}"
        rows.append({"Feature": feature, "Label": f"{feature} = {value}", "Impact": impact})

    return pd.DataFrame(rows).sort_values("Impact", key=abs, ascending=False)


def coefficient_table(model):
    parts = linear_parts(model)
    if parts is None:
        return None
    _, names, coefs = parts
    rows = []
    for name, coef in zip(names, coefs):
        feature, category = _original_feature(name)
        rows.append({"Driver": f"{feature}: {category}" if category else feature, "Coefficient": float(coef)})
    return pd.DataFrame(rows)


def demo_portfolio(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Synthetic customers for demos and screenshots. Not real data."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerID": [f"DEMO-{i + 1:04d}" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
    })
    df["MultipleLines"] = np.where(
        df["PhoneService"] == "No", "No phone service", rng.choice(["Yes", "No"], n)
    )
    for col in INTERNET_FEATURES:
        df[col] = np.where(df["InternetService"] == "No", "No internet service", rng.choice(["Yes", "No"], n))
    base = np.where(df["InternetService"] == "No", 22, np.where(df["InternetService"] == "DSL", 48, 78))
    df["MonthlyCharges"] = (base + rng.normal(0, 8, n)).clip(18.25, 118.75).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    return df[["customerID"] + FEATURE_COLUMNS]


def run_batch(customers: pd.DataFrame, action: str = "Batch scoring") -> pd.DataFrame:
    probabilities = score_customers(model, customers)
    results = customers.copy()
    results["ChurnProbability"] = probabilities
    results["PredictedChurn"] = ["Yes" if p >= threshold else "No" for p in probabilities]
    results["RiskTier"] = [risk_tier(p, threshold) for p in probabilities]
    results["DecisionThreshold"] = threshold
    st.session_state.batch_results = results

    high = int((results["RiskTier"] == "High").sum())
    add_activity({
        "type": "batch",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": action,
        "count": len(results),
        "result": f"{high:,} high-risk customers",
    })
    return results


def add_activity(record: dict) -> None:
    st.session_state.activity.insert(0, record)
    st.session_state.activity = st.session_state.activity[:100]


# ============================================================
# Load model + session state
# ============================================================

try:
    model, threshold = load_assets()
except Exception as exc:
    st.error("ChurnIQ could not load its model files.")
    st.caption("Make sure churn_model.pkl and churn_threshold.pkl are in the same folder as app.py.")
    st.exception(exc)
    st.stop()

if "activity" not in st.session_state:
    st.session_state.activity = []

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## ChurnIQ")
    st.caption("Customer retention intelligence")
    st.divider()

    page = st.radio(
        "Workspace",
        PAGES,
        format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Prediction engine**")
    st.caption("🟢 Model loaded")
    st.caption(f"Decision threshold: {threshold:.1%}")
    st.caption(f"Input features: {len(FEATURE_COLUMNS)}")
    st.divider()
    st.caption("Built with Streamlit, scikit-learn and Plotly.")

# ============================================================
# Overview
# ============================================================

if page == "Overview":
    hero(
        "Customer retention workspace",
        "Know who is about to leave, and what to do about it.",
        "Score a single customer or a whole portfolio, see the risk behind each "
        "prediction, and test retention actions before you take them.",
        ["Churn probability", "Risk tiers", "What-if simulator", "Batch scoring"],
    )

    activity = st.session_state.activity
    singles = [a for a in activity if a.get("type") == "single"]
    batches = [a for a in activity if a.get("type") == "batch"]
    total_scored = sum(a.get("count", 0) for a in batches) + len(singles)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Customers scored", f"{total_scored:,}", "This session")
    with c2:
        kpi("Single assessments", f"{len(singles)}", "Individual accounts", "#54c7ec")
    with c3:
        kpi("Batch runs", f"{len(batches)}", "Portfolio files", "#48d6a0")
    with c4:
        kpi("Decision threshold", f"{threshold:.1%}", "High-risk cutoff", "#fbbf24")

    st.write("")
    st.subheader("How it works")
    a, b, c = st.columns(3)
    with a:
        step_card("1. Provide customer data", "Fill in one profile or upload a CSV with the 19 model input columns.")
    with b:
        step_card("2. Get a risk estimate", "The model returns a churn probability, compared against the saved decision threshold.")
    with c:
        step_card("3. Act on it", "Review retention signals, simulate changes, and export a prioritized list.")

    results = st.session_state.get("batch_results")
    if results is not None:
        st.write("")
        st.subheader("Latest portfolio snapshot")
        left, right = st.columns([1, 1.4])
        counts = results["RiskTier"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        with left:
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=0.65,
                marker={"colors": [TIER_COLORS[t] for t in counts.index]}, sort=False,
            ))
            st.plotly_chart(style_fig(fig, 280), width="stretch")
        with right:
            m1, m2 = st.columns(2)
            high = int((results["RiskTier"] == "High").sum())
            with m1:
                kpi("High-risk customers", f"{high:,}", f"{high / len(results):.1%} of portfolio", "#fb7185")
            with m2:
                kpi("Average churn probability", f"{results['ChurnProbability'].mean():.1%}", "Whole portfolio")
    else:
        st.write("")
        st.info("Nothing scored yet. Open **Single Customer** or **Batch Scoring**, or try the demo.")
        if st.button("Load synthetic demo portfolio (400 customers)", type="primary"):
            run_batch(demo_portfolio(), "Demo portfolio")
            st.rerun()
        st.caption("Demo customers are randomly generated for illustration. They are not real data.")

    st.caption("Session activity lives in memory and resets when the app session restarts.")

# ============================================================
# Single customer
# ============================================================

elif page == "Single Customer":
    hero(
        "Individual assessment",
        "Score a customer",
        "Change any field and the prediction updates instantly. "
        "The result is a model probability, not a guarantee of future behavior.",
    )

    form_col, result_col = st.columns([1.35, 1], gap="large")

    with form_col:
        t_profile, t_sub, t_services, t_billing = st.tabs(
            ["Profile", "Subscription", "Services", "Billing"]
        )

        with t_profile:
            c1, c2 = st.columns(2)
            gender = c1.selectbox("Gender", ["Female", "Male"])
            senior_citizen = c2.selectbox("Senior citizen", ["No", "Yes"])
            partner = c1.selectbox("Has a partner", ["No", "Yes"])
            dependents = c2.selectbox("Has dependents", ["No", "Yes"])

        with t_sub:
            c1, c2 = st.columns(2)
            tenure = c1.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet_service = c1.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            phone_service = c2.selectbox("Phone service", ["Yes", "No"])

        with t_services:
            lines_options = ["No", "Yes"] if phone_service == "Yes" else ["No phone service"]
            net_options = ["No internet service"] if internet_service == "No" else ["No", "Yes"]

            multiple_lines = st.selectbox("Multiple lines", lines_options)
            c1, c2, c3 = st.columns(3)
            online_security = c1.selectbox("Online security", net_options)
            online_backup = c1.selectbox("Online backup", net_options)
            device_protection = c2.selectbox("Device protection", net_options)
            tech_support = c2.selectbox("Tech support", net_options)
            streaming_tv = c3.selectbox("Streaming TV", net_options)
            streaming_movies = c3.selectbox("Streaming movies", net_options)

        with t_billing:
            c1, c2 = st.columns(2)
            paperless_billing = c1.selectbox("Paperless billing", ["Yes", "No"])
            payment_method = c2.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = c1.number_input(
                "Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0
            )
            auto_total = c2.checkbox("Estimate total charges (tenure × monthly)", value=True)
            if auto_total:
                total_charges = float(tenure) * float(monthly_charges)
                c2.caption(f"Total charges: ${total_charges:,.2f}")
            else:
                total_charges = c2.number_input("Total charges ($)", min_value=0.0, value=840.0, step=10.0)

    values = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        customer_df = build_customer_row(values)
        probability = score_customers(model, customer_df)[0]
        prediction = "Yes" if probability >= threshold else "No"
        tier = risk_tier(probability, threshold)
    except Exception as exc:
        st.error("The customer could not be scored.")
        st.exception(exc)
        st.stop()

    with result_col:
        st.markdown("#### Live assessment")
        st.plotly_chart(gauge(probability, threshold), width="stretch")
        st.markdown(
            f'<div style="text-align:center;margin-top:-6px">{tier_pill(tier)}</div>'
            f'<div class="muted" style="text-align:center;margin-top:10px;font-size:13px">'
            f"Predicted churn: <b>{prediction}</b> · threshold {threshold:.1%}</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Save to activity log", type="primary", width="stretch"):
            add_activity({
                "type": "single",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "action": "Single customer",
                "count": 1,
                "prediction": prediction,
                "result": f"{probability:.1%} churn probability ({tier} risk)",
            })
            st.toast("Assessment saved to the activity log.", icon="✅")

    st.divider()
    t_why, t_signals, t_sim, t_export = st.tabs(
        ["Why this score", "Retention signals", "What-if simulator", "Export"]
    )

    with t_why:
        explanation = explain_customer(model, customer_df)
        if explanation is None:
            st.info("Explanations are available for scikit-learn pipelines with a linear classifier.")
        else:
            st.caption(
                "Exact contributions from the logistic regression coefficients. Categories are compared with "
                "the average category of that feature, numeric features with the training average."
            )
            top = explanation.head(8)
            fig = go.Figure(go.Bar(
                x=top["Impact"], y=top["Label"], orientation="h",
                marker={"color": [TIER_COLORS["High"] if v > 0 else TIER_COLORS["Low"] for v in top["Impact"]]},
                text=[f"{v:+.2f}" for v in top["Impact"]], textposition="outside",
            ))
            fig.update_yaxes(autorange="reversed")
            pad = max(abs(top["Impact"])) * 0.3 + 0.05
            fig.update_xaxes(
                title="Impact on churn log-odds (right raises risk, left lowers it)",
                range=[min(top["Impact"].min(), 0) - pad, max(top["Impact"].max(), 0) + pad],
            )
            st.plotly_chart(style_fig(fig, 90 + 42 * len(top)), width="stretch")

    with t_signals:
        signals = risk_signals(values)
        st.caption("Rule-based business heuristics for the retention team. They are not model explanations.")
        if signals:
            for title, action in signals:
                st.markdown(
                    f'<div class="signal"><b>{title}</b><br><span>{action}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No common risk signals found for this profile.")

    with t_sim:
        st.caption(
            "Re-scores this customer with one change at a time. Results show how the model's "
            "estimate shifts. They are associations in the training data, not proven effects."
        )
        try:
            scenarios = run_scenarios(model, values, probability)
        except Exception as exc:
            scenarios = pd.DataFrame()
            st.error("Scenarios could not be scored.")
            st.exception(exc)

        if scenarios.empty:
            st.info("This profile already has the main retention levers in place.")
        else:
            labels = ["Current profile"] + scenarios["Scenario"].tolist()
            probs = [probability * 100] + scenarios["Churn probability (%)"].tolist()
            colors = ["#9b7bfa"] + [
                TIER_COLORS["Low"] if d < 0 else TIER_COLORS["High"] for d in scenarios["Change (pp)"]
            ]
            fig = go.Figure(go.Bar(
                x=probs, y=labels, orientation="h", marker={"color": colors},
                text=[f"{p:.1f}%" for p in probs], textposition="outside",
            ))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[0, max(probs) * 1.2 + 5], title="Churn probability (%)")
            st.plotly_chart(style_fig(fig, 80 + 55 * len(labels)), width="stretch")
            st.dataframe(
                scenarios, hide_index=True, width="stretch",
                column_config={
                    "Churn probability (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Change (pp)": st.column_config.NumberColumn(format="%+.1f pp"),
                },
            )

    with t_export:
        export_df = customer_df.copy()
        export_df["ChurnProbability"] = probability
        export_df["PredictedChurn"] = prediction
        export_df["RiskTier"] = tier
        export_df["DecisionThreshold"] = threshold
        st.dataframe(export_df.T.rename(columns={0: "Value"}).astype(str), width="stretch")
        st.download_button(
            "Download this assessment (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="churniq_customer_assessment.csv",
            mime="text/csv",
        )

    st.caption("Use this as decision support, not as a standalone decision.")

# ============================================================
# Batch scoring
# ============================================================

elif page == "Batch Scoring":
    hero(
        "Portfolio risk assessment",
        "Score a customer file",
        "Upload a CSV with one row per customer. ChurnIQ adds probability, "
        "prediction and risk tier columns and ranks who to contact first.",
    )

    with st.expander("CSV requirements and templates", expanded=False):
        st.write(
            "The file must include all 19 input columns (case-sensitive) and no churn target column. "
            "`customerID` and other extra columns are kept in the output."
        )
        st.code(", ".join(FEATURE_COLUMNS), language="text")
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download empty template",
            data=pd.DataFrame(columns=FEATURE_COLUMNS).to_csv(index=False).encode("utf-8"),
            file_name="churniq_input_template.csv", mime="text/csv", width="stretch",
        )
        d2.download_button(
            "Download sample with 3 example rows",
            data=sample_data().to_csv(index=False).encode("utf-8"),
            file_name="churniq_sample_customers.csv", mime="text/csv", width="stretch",
        )

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            customers = prepare_batch(pd.read_csv(uploaded_file))
            missing = [c for c in FEATURE_COLUMNS if c not in customers.columns]

            if missing:
                st.error("The file is missing required columns: " + ", ".join(missing))
                st.caption("Check spelling and capitalization against the required column list.")
            elif customers.empty:
                st.warning("The uploaded CSV contains no customer rows.")
            else:
                null_cells = int(customers[FEATURE_COLUMNS].isna().sum().sum())
                st.write(f"Rows detected: **{len(customers):,}**")
                if null_cells:
                    st.warning(f"{null_cells:,} empty cells found in the model columns. Scoring may fail or be unreliable.")
                st.dataframe(customers.head(10), hide_index=True, width="stretch")

                if st.button("Score uploaded customers", type="primary", width="stretch"):
                    try:
                        with st.spinner("Scoring customers..."):
                            run_batch(customers)
                        st.success("Batch scoring completed.")
                    except Exception as exc:
                        st.error("Batch scoring failed.")
                        st.exception(exc)
        except Exception as exc:
            st.error("The CSV file could not be read.")
            st.exception(exc)

    if st.button("Or try a synthetic demo portfolio (400 customers)"):
        run_batch(demo_portfolio(), "Demo portfolio")
        st.rerun()

    results = st.session_state.get("batch_results")
    if results is not None:
        st.divider()
        st.subheader("Portfolio results")

        high_df = results[results["RiskTier"] == "High"]
        revenue_at_risk = float(high_df["MonthlyCharges"].sum())

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Customers scored", f"{len(results):,}")
        with k2:
            kpi("High-risk customers", f"{len(high_df):,}", f"{len(high_df) / len(results):.1%} of portfolio", "#fb7185")
        with k3:
            kpi("Monthly revenue at risk", f"${revenue_at_risk:,.0f}", "Sum of high-risk monthly charges", "#fbbf24")
        with k4:
            kpi("Average churn probability", f"{results['ChurnProbability'].mean():.1%}", "", "#54c7ec")

        st.write("")
        t_dist, t_priority, t_all = st.tabs(["Distribution", "Priority list", "All results"])

        with t_dist:
            c1, c2 = st.columns([1.5, 1])
            with c1:
                fig = go.Figure(go.Histogram(
                    x=results["ChurnProbability"] * 100, nbinsx=25,
                    marker={"color": "#9b7bfa", "line": {"color": "#0a0f1d", "width": 1}},
                ))
                fig.add_vline(
                    x=threshold * 100, line_dash="dash", line_color="white",
                    annotation_text="Threshold", annotation_font_color="white",
                )
                fig.update_xaxes(title="Churn probability (%)")
                fig.update_yaxes(title="Customers")
                st.plotly_chart(style_fig(fig, 340), width="stretch")
            with c2:
                counts = results["RiskTier"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
                fig = go.Figure(go.Pie(
                    labels=counts.index, values=counts.values, hole=0.62, sort=False,
                    marker={"colors": [TIER_COLORS[t] for t in counts.index]},
                ))
                st.plotly_chart(style_fig(fig, 340), width="stretch")

            by_contract = results.groupby("Contract")["ChurnProbability"].mean().mul(100).reset_index()
            fig = go.Figure(go.Bar(
                x=by_contract["Contract"], y=by_contract["ChurnProbability"],
                marker={"color": "#54c7ec"},
                text=[f"{v:.1f}%" for v in by_contract["ChurnProbability"]], textposition="outside",
            ))
            fig.update_yaxes(title="Average churn probability (%)")
            st.markdown("**Average risk by contract type**")
            st.plotly_chart(style_fig(fig, 280), width="stretch")

        with t_priority:
            f1, f2 = st.columns(2)
            tiers = f1.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High"])
            contracts = f2.multiselect(
                "Contract", sorted(results["Contract"].dropna().unique()),
                default=sorted(results["Contract"].dropna().unique()),
            )
            view = results[results["RiskTier"].isin(tiers) & results["Contract"].isin(contracts)].copy()
            view = view.sort_values("ChurnProbability", ascending=False)
            view.insert(0, "Risk (%)", view["ChurnProbability"] * 100)

            id_cols = [c for c in ("customerID",) if c in view.columns]
            show_cols = id_cols + ["Risk (%)", "RiskTier", "Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod"]
            st.caption(f"{len(view):,} customers match, sorted by risk.")
            st.dataframe(
                view[show_cols].head(200), hide_index=True, width="stretch",
                column_config={
                    "Risk (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                },
            )
            st.download_button(
                "Download this priority list (CSV)",
                data=view.drop(columns=["Risk (%)"]).to_csv(index=False).encode("utf-8"),
                file_name="churniq_priority_list.csv", mime="text/csv",
            )

        with t_all:
            st.dataframe(results, hide_index=True, width="stretch")
            st.download_button(
                "Download all scored customers (CSV)",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="churniq_scored_customers.csv", mime="text/csv", width="stretch",
            )

# ============================================================
# Model performance
# ============================================================

elif page == "Model Performance":
    hero(
        "Model evaluation",
        "Model performance",
        "Upload a labeled CSV to measure how well the model separates churners from "
        "loyal customers at the saved decision threshold.",
    )
    st.info(
        "Use data the model was not trained on, such as your test split. "
        "Metrics on training data look better than they really are."
    )
    eval_file = st.file_uploader("Upload labeled CSV (19 input columns plus a Churn column)", type=["csv"], key="eval_file")

    if eval_file is None:
        st.caption("Churn can be Yes/No or 1/0. Export your test split from the notebook with `X_test.assign(Churn=y_test).to_csv(...)`.")
    else:
        try:
            data = prepare_batch(pd.read_csv(eval_file))
            missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
            if "Churn" not in data.columns:
                st.error("The file needs a Churn column.")
            elif missing:
                st.error("The file is missing required columns: " + ", ".join(missing))
            else:
                labels = (
                    data["Churn"].astype(str).str.strip().str.lower()
                    .map({"yes": 1, "1": 1, "true": 1, "no": 0, "0": 0, "false": 0})
                )
                valid = labels.notna()
                if not valid.all():
                    st.warning(f"{int((~valid).sum())} rows with unrecognized Churn values were skipped.")
                data, y = data[valid], labels[valid].astype(int).to_numpy()

                if len(set(y)) < 2:
                    st.error("The file needs both churned and retained customers.")
                else:
                    probs = np.array(score_customers(model, data))
                    pred = (probs >= threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

                    cols = st.columns(5)
                    metrics = [
                        ("ROC-AUC", f"{roc_auc_score(y, probs):.3f}", "Ranking quality", "#9b7bfa"),
                        ("Recall", f"{recall_score(y, pred, zero_division=0):.1%}", "Churners caught", "#48d6a0"),
                        ("Precision", f"{precision_score(y, pred, zero_division=0):.1%}", "Flags that were right", "#54c7ec"),
                        ("F1 score", f"{f1_score(y, pred, zero_division=0):.3f}", "Precision and recall", "#fbbf24"),
                        ("Accuracy", f"{accuracy_score(y, pred):.1%}", f"{len(y):,} customers", "#fb7185"),
                    ]
                    for col, (label, value, sub, accent) in zip(cols, metrics):
                        with col:
                            kpi(label, value, sub, accent)

                    st.write("")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**ROC curve**")
                        fpr, tpr, _ = roc_curve(y, probs)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line={"color": "#9b7bfa", "width": 3}))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"color": "#51608a", "dash": "dash"}))
                        fig.add_trace(go.Scatter(
                            x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode="markers", name="Current threshold",
                            marker={"color": "white", "size": 12, "line": {"color": "#9b7bfa", "width": 2}},
                        ))
                        fig.update_xaxes(title="False positive rate")
                        fig.update_yaxes(title="True positive rate")
                        st.plotly_chart(style_fig(fig, 360), width="stretch")
                    with c2:
                        st.markdown("**Confusion matrix**")
                        matrix = [[tn, fp], [fn, tp]]
                        fig = go.Figure(go.Heatmap(
                            z=matrix, x=["Predicted: stay", "Predicted: churn"], y=["Actual: stay", "Actual: churn"],
                            text=[[f"{v:,}" for v in row] for row in matrix], texttemplate="%{text}",
                            textfont={"size": 22}, colorscale=[[0, "#18223a"], [1, "#7555ed"]], showscale=False,
                        ))
                        fig.update_yaxes(autorange="reversed")
                        st.plotly_chart(style_fig(fig, 360), width="stretch")

                    st.markdown("**Threshold trade-off**")
                    grid = np.linspace(0.05, 0.95, 91)
                    sweep = pd.DataFrame({
                        "Threshold": grid,
                        "Precision": [precision_score(y, (probs >= t).astype(int), zero_division=0) for t in grid],
                        "Recall": [recall_score(y, (probs >= t).astype(int), zero_division=0) for t in grid],
                        "F1": [f1_score(y, (probs >= t).astype(int), zero_division=0) for t in grid],
                    })
                    fig = go.Figure()
                    for name, color in [("Precision", "#54c7ec"), ("Recall", "#48d6a0"), ("F1", "#fbbf24")]:
                        fig.add_trace(go.Scatter(x=sweep["Threshold"], y=sweep[name], mode="lines", name=name, line={"color": color, "width": 3}))
                    fig.add_vline(x=threshold, line_dash="dash", line_color="white", annotation_text="Saved threshold", annotation_font_color="white")
                    fig.update_xaxes(title="Decision threshold", tickformat=".0%")
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(style_fig(fig, 340), width="stretch")
                    st.caption("Lower thresholds catch more churners but flag more customers who would have stayed.")
        except Exception as exc:
            st.error("The evaluation file could not be processed.")
            st.exception(exc)

# ============================================================
# Activity
# ============================================================

elif page == "Activity":
    hero(
        "Session workspace",
        "Scoring activity",
        "Actions completed during this app session. This is temporary activity, not a permanent audit log.",
    )

    activity = st.session_state.activity
    if not activity:
        st.info("No scoring activity yet. Save a single assessment or run a batch to see it here.")
    else:
        table = pd.DataFrame([
            {
                "Time": a.get("time", ""), "Action": a.get("action", ""),
                "Customers": a.get("count", 1), "Result": a.get("result", ""),
            }
            for a in activity
        ])
        st.dataframe(table, hide_index=True, width="stretch")
        c1, c2 = st.columns([1, 4])
        c1.download_button(
            "Export log (CSV)", data=table.to_csv(index=False).encode("utf-8"),
            file_name="churniq_activity.csv", mime="text/csv",
        )
        if c2.button("Clear session activity"):
            st.session_state.activity = []
            st.rerun()

# ============================================================
# Model information
# ============================================================

elif page == "Model Information":
    hero(
        "Model transparency",
        "Model information",
        "The active model setup, its expected input schema, and what the decision threshold means.",
    )

    steps = getattr(model, "steps", None)
    model_name = steps[-1][1].__class__.__name__ if steps else model.__class__.__name__

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Status", "Loaded", "", "#48d6a0")
    with c2:
        kpi("Algorithm", model_name, "Final estimator")
    with c3:
        kpi("Input features", str(len(FEATURE_COLUMNS)), "", "#54c7ec")
    with c4:
        kpi("Decision threshold", f"{threshold:.1%}", "", "#fbbf24")

    st.write("")
    st.subheader("Prediction flow")
    a, b, c, d = st.columns(4)
    with a:
        step_card("Customer input", "Profile, services and billing fields.")
    with b:
        step_card("Preprocessing", "Saved transformations inside the model pipeline.")
    with c:
        step_card("Classifier", "Outputs a churn probability.")
    with d:
        step_card("Threshold", "Compared with the saved cutoff to set the class.")

    if steps:
        st.write("")
        st.subheader("Pipeline steps")
        st.dataframe(
            pd.DataFrame({"Step": [n for n, _ in steps], "Component": [s.__class__.__name__ for _, s in steps]}),
            hide_index=True, width="stretch",
        )

    st.write("")
    st.subheader("Input schema")
    schema = pd.DataFrame(
        [{"Group": group, "Column": col} for group, cols in FEATURE_GROUPS.items() for col in cols]
    )
    st.dataframe(schema, hide_index=True, width="stretch")

    coefs = coefficient_table(model)
    if coefs is not None:
        st.subheader("Strongest drivers")
        st.caption("Logistic regression coefficients. Positive values raise churn risk, negative values lower it.")
        top = coefs.reindex(coefs["Coefficient"].abs().sort_values(ascending=False).index).head(12)
        fig = go.Figure(go.Bar(
            x=top["Coefficient"], y=top["Driver"], orientation="h",
            marker={"color": [TIER_COLORS["High"] if v > 0 else TIER_COLORS["Low"] for v in top["Coefficient"]]},
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Coefficient (log-odds)")
        st.plotly_chart(style_fig(fig, 430), width="stretch")

    st.subheader("About the threshold")
    st.write(
        f"A probability at or above **{threshold:.1%}** is classified as **Yes**. "
        f"Probabilities between {threshold * 0.6:.1%} and {threshold:.1%} are shown as **Medium** risk "
        "so teams can watch borderline accounts. The threshold is loaded from `churn_threshold.pkl`."
    )
    st.warning(
        "A threshold is a classification setting, not a measure of accuracy. "
        "Add verified held-out test metrics separately if you want to report model performance."
    )

# ============================================================
# Footer
# ============================================================

st.markdown('<div class="footer">ChurnIQ · Customer Retention Intelligence</div>', unsafe_allow_html=True)