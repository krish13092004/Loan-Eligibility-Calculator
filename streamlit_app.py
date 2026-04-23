"""
🏦 Loan Eligibility Prediction — Streamlit Dashboard
=====================================================

A comprehensive, interactive dashboard showcasing:
• 8 ML models for loan eligibility prediction
• Novel SWE (Smart Weighted Ensemble) algorithm
• Dataset exploration, model training, live predictions

Author: Loan Eligibility Project
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, pickle, os
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from SmartWeightedEnsemble import SmartWeightedEnsemble
from HierarchicalBayesianTrustEnsemble import HierarchicalBayesianTrustEnsemble

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for premium look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* -------- global font -------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* -------- sidebar -------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stRadio label { font-weight: 500; }

/* -------- metric cards -------- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%);
    border: 1px solid #667eea33;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 15px rgba(102,126,234,.08);
}
div[data-testid="stMetric"] label { font-weight: 600 !important; }

/* -------- headers -------- */
h1 { background: linear-gradient(90deg,#667eea,#764ba2);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-weight: 800 !important; }
h2, h3 { color: #4a5568 !important; font-weight: 700 !important; }

/* -------- buttons -------- */
.stButton>button {
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    color: white !important; border: none; border-radius: 10px;
    padding: 0.55rem 2rem; font-weight: 600;
    transition: all .3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102,126,234,.35);
}

/* -------- tabs -------- */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 10px 24px;
    font-weight: 600; background: #f7f7fa;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important;
    color: white !important;
}

/* -------- info-card -------- */
.info-card {
    background: linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);
    border-radius: 16px; padding: 24px 28px; margin-bottom: 18px;
    border-left: 5px solid #667eea;
}
.info-card h4 { margin: 0 0 8px; color: #2d3748; }
.info-card p  { margin: 0; color: #4a5568; line-height: 1.6; }

/* -------- success/feature boxes -------- */
.feature-box {
    background: white; border-radius: 14px;
    padding: 20px; text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 10px rgba(0,0,0,.04);
    transition: transform .2s;
}
.feature-box:hover { transform: translateY(-4px); box-shadow: 0 8px 25px rgba(0,0,0,.08); }

/* -------- expander -------- */
.streamlit-expanderHeader { font-weight: 600 !important; font-size: 1.05rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: colour palettes
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [
    "#667eea", "#764ba2", "#f093fb", "#f5576c",
    "#4facfe", "#00f2fe", "#43e97b", "#fa709a",
    "#fccb90", "#a18cd1",
]

GRADIENT_BG = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#2d3748"),
)

def styled_plotly(fig, height=420):
    fig.update_layout(
        **GRADIENT_BG,
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,.7)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=12),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    csv_path = os.path.join(os.path.dirname(__file__), "LoanData.csv")
    raw = pd.read_csv(csv_path)
    data = raw.copy()

    # Missing-value imputation
    data["Gender"].fillna(data["Gender"].mode()[0], inplace=True)
    data["Married"].fillna(data["Married"].mode()[0], inplace=True)
    data["Dependents"].fillna(data["Dependents"].mode()[0], inplace=True)
    data["Self_Employed"].fillna(data["Self_Employed"].mode()[0], inplace=True)
    data["LoanAmount"].fillna(data["LoanAmount"].median(), inplace=True)
    data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0], inplace=True)
    data["Credit_History"].fillna(0, inplace=True)

    # Feature engineering
    data["Total_Income"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
    data["Loan_Income_Ratio"] = data["LoanAmount"] / (data["Total_Income"] + 1)

    # Encode
    le_map = {}
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_map[col] = le

    X = data.drop(["Loan_Status", "Loan_ID", "ApplicantIncome", "CoapplicantIncome"], axis=1, errors="ignore")
    y = data["Loan_Status"]

    # Dependents column: convert '3+' -> 3 and handle any remaining strings
    X["Dependents"] = X["Dependents"].astype(str).replace("3+", "3")
    X["Dependents"] = pd.to_numeric(X["Dependents"], errors='coerce').fillna(0).astype(int)

    # Ensure all columns in X are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Ensure y is numeric
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

    return raw, data, X, y, le_map


@st.cache_resource
def train_all_models(_X, _y, power=2.0):
    """Train 8 base models + SWE and return results."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        _X, _y, test_size=0.3, random_state=42, stratify=_y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=12, p=1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        )

    # --- SWE ---
    swe_models = deepcopy(models)
    swe = SmartWeightedEnsemble(swe_models, power=power, verbose=0)
    swe.fit(X_train.values, y_train.values, X_val.values, y_val.values)
    swe_pred, swe_conf = swe.predict(X_test.values, return_confidence=True)
    swe_proba = swe.predict_proba(X_test.values)

    hbte_models = deepcopy(models)
    hbte = HierarchicalBayesianTrustEnsemble(hbte_models, beta=power, verbose=0)
    hbte.fit(X_train.values, y_train.values, X_val.values, y_val.values)
    hbte_pred, hbte_conf = hbte.predict(X_test.values, return_confidence=True)
    hbte_proba = hbte.predict_proba(X_test.values)

    # Individual model results (use SWE's trained models for evaluation)
    individual = {}
    for name in swe_models:
        model = swe.models[name]
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        individual[name] = dict(
            accuracy=accuracy_score(y_test, pred),
            auc=roc_auc_score(y_test, proba[:, 1]),
            predictions=pred,
            proba=proba,
        )

    results = dict(
        individual=individual,
        swe=dict(
            accuracy=accuracy_score(y_test, swe_pred),
            auc=roc_auc_score(y_test, swe_proba[:, 1]),
            predictions=swe_pred,
            proba=swe_proba,
            confidence=swe_conf,
            model=swe,
        ),
        hbte=dict(
            accuracy=accuracy_score(y_test, hbte_pred),
            auc=roc_auc_score(y_test, hbte_proba[:, 1]),
            predictions=hbte_pred,
            proba=hbte_proba,
            confidence=hbte_conf,
            model=hbte,
        ),
        splits=dict(X_train=X_train, X_val=X_val, X_test=X_test,
                     y_train=y_train, y_val=y_val, y_test=y_test),
        models=models,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Navigation")
    page = st.radio(
        "Go to",
        [
            "🏠 Dashboard",
            "📊 Dataset Explorer",
            "🤖 Model Training",
            "🔮 Live Prediction",
            "📐 Algorithm Deep-Dive",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚙️ Hyperparameters")
    power = st.slider("SWE / HBTE Power (β)", 1.0, 5.0, 2.0, 0.5)
    st.markdown("---")
    st.caption("© 2026 · Loan Eligibility Project")


# ─────────────────────────────────────────────────────────────────────────────
# Load data + models
# ─────────────────────────────────────────────────────────────────────────────
raw, data, X, y, le_map = load_and_preprocess()
results = train_all_models(X, y, power=power)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# 🏦 Loan Eligibility Prediction")
    st.markdown("##### Intelligent Credit Decision System using Machine Learning & Novel Ensemble Algorithm")
    st.markdown("")

    # ── KPI row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    best_ind_name = max(results["individual"], key=lambda k: results["individual"][k]["accuracy"])
    best_ind_acc = results["individual"][best_ind_name]["accuracy"]

    c1.metric("Dataset Size", f"{len(raw)} rows")
    c2.metric("Features", f"{X.shape[1]}")
    c3.metric("Best Individual", f"{best_ind_acc:.2%}")
    c4.metric("SWE Accuracy", f"{results['swe']['accuracy']:.2%}")
    c5.metric("HBTE Accuracy", f"{results['hbte']['accuracy']:.2%}")

    st.markdown("")

    # ── Accuracy comparison chart ──
    names = list(results["individual"].keys()) + ["SWE ⭐", "HBTE 🏛️"]
    accs = [results["individual"][n]["accuracy"] for n in results["individual"]] + [
        results["swe"]["accuracy"], results["hbte"]["accuracy"]
    ]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(results["individual"]))] + ["#f5576c", "#ab47bc"]

    fig = go.Figure(go.Bar(
        x=names, y=accs,
        marker=dict(color=colors, line=dict(width=0), cornerradius=6),
        text=[f"{a:.2%}" for a in accs],
        textposition="outside",
        textfont=dict(size=13, family="Inter", color="#2d3748"),
    ))
    fig.update_layout(
        title=dict(text="📊 Model Accuracy Comparison", font=dict(size=20)),
        yaxis=dict(title="Accuracy", range=[0.55, max(accs) + 0.08], gridcolor="#edf2f7"),
        xaxis=dict(title=""),
    )
    styled_plotly(fig, 440)
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature boxes ──
    st.markdown("### ✨ Key Features")
    f1, f2, f3 = st.columns(3)
    features = [
        ("🧠", "8 ML Algorithms", "LogReg, KNN, DT, RF, SVM, XGBoost, AdaBoost, Naïve Bayes"),
        ("⚡", "2 Novel Ensembles", "SWE (Power-Based Voting) & HBTE (Hierarchical Bayesian Trust)"),
        ("🎯", "Live Prediction", "Enter applicant details and get real-time eligibility + confidence"),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3], features):
        col.markdown(
            f"""<div class="feature-box">
                    <h2 style="margin:0; font-size:2.2rem;">{icon}</h2>
                    <h4 style="margin:8px 0 4px; color:#2d3748;">{title}</h4>
                    <p style="font-size:.88rem; color:#718096;">{desc}</p>
                </div>""",
            unsafe_allow_html=True,
        )



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":
    st.markdown("# 📊 Dataset Explorer")
    st.markdown("##### Explore the Loan Eligibility dataset — distributions, correlations & insights")
    st.markdown("")

    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Distributions", "🔗 Correlations"])

    with tab1:
        st.dataframe(raw.head(50), use_container_width=True, height=400)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(raw))
        c2.metric("Missing Values", int(raw.isnull().sum().sum()))
        c3.metric("Approved %", f"{(raw['Loan_Status']=='Y').mean():.1%}")

    with tab2:
        col_choice = st.selectbox("Select feature", raw.columns.drop(["Loan_ID"]))
        if raw[col_choice].dtype == "object":
            counts = raw[col_choice].value_counts()
            fig = px.pie(
                values=counts.values, names=counts.index, title=f"Distribution of {col_choice}",
                color_discrete_sequence=PALETTE, hole=0.45,
            )
        else:
            fig = px.histogram(
                raw, x=col_choice, color="Loan_Status", barmode="overlay",
                color_discrete_sequence=[PALETTE[0], PALETTE[3]],
                title=f"Distribution of {col_choice} by Loan Status",
            )
        styled_plotly(fig, 440)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        numeric = data.select_dtypes(include=np.number)
        corr = numeric.corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
            title="Feature Correlation Matrix",
        )
        styled_plotly(fig, 520)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("# 🤖 Model Training & Comparison")
    st.markdown("##### Side-by-side evaluation of all models on the test set")
    st.markdown("")

    y_test = results["splits"]["y_test"]
    X_test = results["splits"]["X_test"]

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Accuracy", "📉 ROC Curves", "🎯 Confusion Matrix", "⚖️ Model Weights"])

    # ── Tab 1: accuracy table + chart ──
    with tab1:
        rows = []
        for name, r in results["individual"].items():
            rows.append(dict(Model=name, Accuracy=r["accuracy"], AUC=r["auc"], Type="Individual"))
        rows.append(dict(Model="SWE ⭐", Accuracy=results["swe"]["accuracy"], AUC=results["swe"]["auc"], Type="Ensemble"))
        rows.append(dict(Model="HBTE 🏛️", Accuracy=results["hbte"]["accuracy"], AUC=results["hbte"]["auc"], Type="Ensemble"))
        df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        df.index += 1

        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(
                df.style.format({"Accuracy": "{:.4f}", "AUC": "{:.4f}"})
                .background_gradient(subset=["Accuracy"], cmap="YlGn"),
                use_container_width=True,
            )
        with c2:
            fig = px.bar(
                df, x="Model", y="Accuracy", color="Type",
                color_discrete_map={"Individual": PALETTE[0], "Ensemble": "#f5576c"},
                text_auto=".3f",
                title="Accuracy Ranking",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis=dict(range=[0.5, df["Accuracy"].max() + 0.08]))
            styled_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: ROC curves ──
    with tab2:
        fig = go.Figure()
        for i, (name, r) in enumerate(results["individual"].items()):
            fpr, tpr, _ = roc_curve(y_test, r["proba"][:, 1])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f'{name} (AUC={r["auc"]:.3f})',
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
        # SWE
        fpr, tpr, _ = roc_curve(y_test, results["swe"]["proba"][:, 1])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f'SWE (AUC={results["swe"]["auc"]:.3f})',
                                 line=dict(color="#f5576c", width=3, dash="dash")))
        # HBTE
        fpr, tpr, _ = roc_curve(y_test, results["hbte"]["proba"][:, 1])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f'HBTE (AUC={results["hbte"]["auc"]:.3f})',
                                 line=dict(color="#ab47bc", width=3, dash="dash")))
        # Diagonal
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color="grey", dash="dot"), showlegend=False))
        fig.update_layout(
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        styled_plotly(fig, 500)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: confusion matrices ──
    with tab3:
        model_pick = st.selectbox(
            "Select model for confusion matrix",
            list(results["individual"].keys()) + ["SWE", "HBTE"],
        )
        if model_pick in results["individual"]:
            preds = results["individual"][model_pick]["predictions"]
        elif model_pick == "SWE":
            preds = results["swe"]["predictions"]
        else:
            preds = results["hbte"]["predictions"]

        cm = confusion_matrix(y_test, preds)
        fig = px.imshow(
            cm, text_auto=True,
            x=["Rejected", "Approved"], y=["Rejected", "Approved"],
            color_continuous_scale=[[0, "#f7f7fa"], [1, PALETTE[0]]],
            labels=dict(x="Predicted", y="Actual"),
            title=f"Confusion Matrix — {model_pick}",
        )
        styled_plotly(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

        report = classification_report(y_test, preds, target_names=["Rejected", "Approved"], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)

    # ── Tab 4: model weights ──
    with tab4:
        st.markdown("### SWE Model Weights")
        swe_model = results["swe"]["model"]
        w_df = pd.DataFrame({
            "Model": list(swe_model.weights_.keys()),
            "Weight": list(swe_model.weights_.values()),
        }).sort_values("Weight", ascending=True)

        fig = px.bar(
            w_df, x="Weight", y="Model", orientation="h",
            color="Weight", color_continuous_scale="Viridis",
            title="SWE Power-Weighted Model Contributions",
        )
        styled_plotly(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📖 How Weights Are Calculated")
        st.latex(r"w_k = \frac{\alpha_k^{\beta}}{\sum_{j=1}^{K} \alpha_j^{\beta}}")
        st.markdown("""
        - **α_k** = validation accuracy of model k
        - **β** = power parameter (currently set to {:.1f} via sidebar)
        - Higher β → more weight to the best model, less to weaker ones
        """.format(power))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.markdown("# 🔮 Live Loan Eligibility Prediction")
    st.markdown("##### Enter applicant details and get instant eligibility decisions with confidence scores")
    st.markdown("")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with c2:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            applicant_income = st.number_input("Applicant Income ($)", 0, 100000, 5000, 500)
            coapplicant_income = st.number_input("Co-applicant Income ($)", 0, 100000, 0, 500)
        with c3:
            loan_amount = st.number_input("Loan Amount ($K)", 0, 1000, 150, 10)
            loan_term = st.selectbox("Loan Term (days)", [360, 180, 480, 240, 120, 60, 36, 12])
            credit_history = st.selectbox("Credit History", [1.0, 0.0])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("🚀  Predict Eligibility", use_container_width=True)

    if submitted:
        # Encode input
        gender_enc = 1 if gender == "Male" else 0
        married_enc = 1 if married == "Yes" else 0
        dep_enc = 3 if dependents == "3+" else int(dependents)
        edu_enc = 0 if education == "Graduate" else 1
        se_enc = 1 if self_employed == "Yes" else 0
        pa_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
        pa_enc = pa_map[property_area]
        total_income = applicant_income + coapplicant_income
        lir = loan_amount / (total_income + 1)

        feature_order = X.columns.tolist()
        input_dict = {
            "Gender": gender_enc, "Married": married_enc, "Dependents": dep_enc,
            "Education": edu_enc, "Self_Employed": se_enc,
            "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history, "Property_Area": pa_enc,
            "Total_Income": total_income, "Loan_Income_Ratio": lir,
        }
        input_df = pd.DataFrame([input_dict])[feature_order]
        x_arr = input_df.values

        swe_model = results["swe"]["model"]

        swe_pred, swe_conf = swe_model.predict(x_arr, return_confidence=True)
        swe_proba = swe_model.predict_proba(x_arr)[0]

        st.markdown("---")
        st.markdown("### 🏆 Prediction Results")

        swe_pred_val = swe_pred[0]
        swe_conf_val = swe_conf[0]
        swe_status = "✅ APPROVED" if swe_pred_val == 1 else "❌ REJECTED"
        swe_colour = "#43e97b" if swe_pred_val == 1 else "#f5576c"

        hbte_model = results["hbte"]["model"]
        hbte_info = hbte_model.explain_prediction(x_arr, 0)
        hbte_pred_val = 1 if hbte_info["final_prediction"] == "Approved" else 0
        hbte_status = "✅ APPROVED" if hbte_pred_val == 1 else "❌ REJECTED"
        hbte_colour = "#43e97b" if hbte_pred_val == 1 else "#f5576c"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div style="background:linear-gradient(135deg,{swe_colour}22,{swe_colour}11);
                        border-radius:16px;padding:28px;text-align:center;
                        border:2px solid {swe_colour}55; height: 100%;">
                    <h3 style="margin:0;color:#2d3748;">⚡ SWE Prediction</h3>
                    <h1 style="margin:10px 0;background:none;-webkit-text-fill-color:{swe_colour};
                        font-size:2.2rem;">{swe_status}</h1>
                    <p style="font-size:1.1rem;color:#4a5568;">
                        Confidence: <b>{swe_conf_val:.1%}</b><br>
                        P(Approved): <b>{swe_proba[1]:.1%}</b>
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div style="background:linear-gradient(135deg,{hbte_colour}22,{hbte_colour}11);
                        border-radius:16px;padding:28px;text-align:center;
                        border:2px solid {hbte_colour}55; height: 100%;">
                    <h3 style="margin:0;color:#2d3748;">🏛️ HBTE Prediction</h3>
                    <h1 style="margin:10px 0;background:none;-webkit-text-fill-color:{hbte_colour};
                        font-size:2.2rem;">{hbte_status}</h1>
                    <p style="font-size:1.1rem;color:#4a5568;">
                        Confidence (Γ): <b>{hbte_info['confidence_gamma']:.1%}</b><br>
                        {hbte_info['tier_label']}
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

        # Individual model votes
        st.markdown("### 🗳️ Individual Model Votes")
        vote_data = []
        for name in results["models"]:
            model = swe_model.models[name]
            p = model.predict(x_arr)[0]
            prob = model.predict_proba(x_arr)[0]
            vote_data.append(dict(
                Model=name,
                Vote="Approved" if p == 1 else "Rejected",
                Probability=prob[1],
            ))
        vote_df = pd.DataFrame(vote_data)

        fig = px.bar(
            vote_df, x="Model", y="Probability", color="Vote",
            color_discrete_map={"Approved": "#43e97b", "Rejected": "#f5576c"},
            title="Individual Model Probability of Approval",
            text_auto=".2f",
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="grey", annotation_text="Decision Boundary")
        fig.update_traces(textposition="outside")
        styled_plotly(fig, 420)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ALGORITHM DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Algorithm Deep-Dive":
    st.markdown("# 📐 Algorithm Deep-Dive")
    st.markdown("##### Understanding the mathematics behind the novel ensembles")
    st.markdown("")

    tab_swe, tab_hbte = st.tabs(["⚡ Smart Weighted Ensemble (SWE)", "🏛️ Hierarchical Bayesian Trust (HBTE)"])

    with tab_swe:
        st.markdown("""
        <div class="info-card">
            <h4>⚡ Smart Weighted Ensemble (SWE)</h4>
            <p>A novel, interpretable ensemble that combines 8 classifiers using
            <b>performance-based dynamic weighting</b> and
            <b>confidence quantification</b>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔢 The 3 Core Formulas")

    c1, c2, c3 = st.columns(3)
    c1.latex(r"w_k = \frac{\alpha_k^{\beta}}{\sum_{j=1}^{K} \alpha_j^{\beta}}")
    c1.caption("**Formula 1 — Weight Calculation**")

    c2.latex(r"P(\text{Approved}|x) = \sum_{k=1}^{K} w_k \cdot P_k(\text{Approved}|x)")
    c2.caption("**Formula 2 — Weighted Prediction**")

    c3.latex(r"\text{Conf} = P_{\max} \times \frac{|\text{Agreed}|}{K}")
    c3.caption("**Formula 3 — Confidence Score**")

    st.markdown("---")

    # ── How SWE works step by step ──
    st.markdown("### 🔄 How SWE Works — Step by Step")

    st.markdown("""
    **Step 1 — Train** all 8 base models on the training data.

    **Step 2 — Evaluate** each model on the validation set to get accuracy scores.

    **Step 3 — Calculate weights** using the power formula:
    - Square each model's accuracy (with β=2)
    - Divide by the sum of all squared accuracies
    - Better models get proportionally MORE weight

    **Step 4 — Predict** by combining all models' probabilities using their weights.

    **Step 5 — Confidence** is computed as: max probability × model agreement fraction.
    """)

    st.markdown("---")
    st.markdown("### 📊 How Power (β) Affects Weights")

    # Interactive power demo
    demo_accs = np.array([0.78, 0.82, 0.75, 0.88, 0.80, 0.86, 0.79, 0.73])
    demo_names = ["LR", "KNN", "DT", "RF", "SVM", "XGB", "Ada", "NB"]
    powers = np.arange(0.5, 5.5, 0.5)
    frames = []
    for p in powers:
        w = demo_accs ** p
        w = w / w.sum()
        for n, ww in zip(demo_names, w):
            frames.append(dict(Model=n, Weight=ww, Power=f"β={p:.1f}"))
    pwr_df = pd.DataFrame(frames)
    fig = px.bar(
        pwr_df, x="Model", y="Weight", animation_frame="Power",
        color="Model", color_discrete_sequence=PALETTE,
        title="Effect of Power Parameter on Model Weights",
    )
    fig.update_layout(yaxis=dict(range=[0, 0.35]))
    styled_plotly(fig, 450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Why SWE is Novel ──
    st.markdown("### 🆚 Comparison with Other Ensemble Methods")

    comparison_df = pd.DataFrame({
        "Method": ["Simple Voting", "Soft Voting", "Weighted Voting", "Stacking", "SWE (Ours) ⭐"],
        "Weights": ["Equal", "Equal", "Fixed", "Learned (black-box)", "Dynamic (power-based)"],
        "Confidence Score": ["❌", "❌", "❌", "❌", "✅"],
        "Explainable": ["✅", "✅", "✅", "❌", "✅"],
        "Novel": ["❌", "❌", "❌", "❌", "✅"],
    })
    st.table(comparison_df)

    st.markdown("---")
    st.markdown("### 🔑 What Makes SWE Novel?")
    st.markdown("""
    1. **Power-Weighted Formula** — Uses α² to amplify differences between models.
       Unlike equal voting or fixed weights, SWE dynamically adapts based on validation accuracy.

    2. **Confidence Quantification** — Unique formula combining prediction probability
       with inter-model agreement. Most ensemble methods don't provide confidence scores.

        3. **Full Explainability** — Every prediction can be traced back to individual
           model votes and their weights. Critical for financial applications like loan approval.
        """)
        
    with tab_hbte:
        st.markdown("""
        <div class="info-card" style="border-left-color: #ab47bc;">
            <h4>🏛️ Hierarchical Bayesian Trust Ensemble (HBTE)</h4>
            <p>An advanced ensemble that uses <b>Bayesian posterior updating</b> to learn model trust over time, and a <b>hierarchical 3-tier system</b> based on Shannon entropy confidence.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔢 The 5 Formulas")
        c1, c2 = st.columns(2)
        c1.latex(r"\alpha_k = \frac{\text{acc}_k^{\beta}}{\sum_j \text{acc}_j^{\beta}}")
        c1.caption("**Formula 1 — Initial Prior Trust**")
        c2.latex(r"\tau_k(t+1) = \frac{s_k + \alpha_k \cdot t_0}{s_k + f_k + t_0}")
        c2.caption("**Formula 2 — Bayesian Trust Update**")
        
        c3, c4 = st.columns(2)
        c3.latex(r"C(x) = 1 - \frac{H(x)}{\log(2)}")
        c3.caption("**Formula 3 — Information Confidence (using Shannon Entropy)**")
        c4.latex(r"\Gamma(x) = \lambda \cdot C(x) + (1-\lambda) \cdot A(x)")
        c4.caption("**Formula 4 — Combined Conviction Score**")

        st.markdown("---")
        st.markdown("### 🔄 How HBTE Works — The 3 Tiers")
        st.markdown("""
        Unlike SWE which always uses 8 models, HBTE picks the right number of models based on the **Confidence Score Γ(x)**:
        
        - ⚡ **Tier 1 (Γ ≥ 0.80):** High confidence. Uses **Top 3 models**. Fastest decision.
        - ⚖️ **Tier 2 (0.60 ≤ Γ < 0.80):** Medium confidence. Uses **Top 5 models**. Balanced decision.
        - 🔍 **Tier 3 (Γ < 0.60):** Low confidence. Uses **All 8 models**. Most thorough decision.
        """)
        
        st.markdown("---")
        st.markdown("### 🔑 What Makes HBTE Novel?")
        st.markdown("""
        1. **Bayesian Learning** — Trust isn't fixed. As HBTE sees real loan outcomes, it rewards models that get answers right ($s_k$) and penalizes failure ($f_k$).
        2. **Information-Theoretic Confidence** — Uses rigorous Shannon entropy ($H(x)$) instead of simple probability.
        3. **Fewer Models When Possible** — The tier system mimics human banking: easy loans are auto-approved quickly (top 3 models), complex ones go to committee (8 models).
        """)
