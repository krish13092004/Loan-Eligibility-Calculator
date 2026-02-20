"""Quick scan of power/beta combos to find where one ensemble fails."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from copy import deepcopy
from SmartWeightedEnsemble import SmartWeightedEnsemble
from HierarchicalBayesianTrustEnsemble import HierarchicalBayesianTrustEnsemble

# ── Load & preprocess ───────────────────────────────────────────────────────
data = pd.read_csv("LoanData.csv")
data["Gender"].fillna(data["Gender"].mode()[0], inplace=True)
data["Married"].fillna(data["Married"].mode()[0], inplace=True)
data["Dependents"].fillna(data["Dependents"].mode()[0], inplace=True)
data["Self_Employed"].fillna(data["Self_Employed"].mode()[0], inplace=True)
data["LoanAmount"].fillna(data["LoanAmount"].median(), inplace=True)
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0], inplace=True)
data["Credit_History"].fillna(0, inplace=True)
data["Total_Income"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
data["Loan_Income_Ratio"] = data["LoanAmount"] / (data["Total_Income"] + 1)

le = LabelEncoder()
for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
    data[col] = le.fit_transform(data[col])

X = data.drop(["Loan_Status", "Loan_ID", "ApplicantIncome", "CoapplicantIncome"], axis=1, errors="ignore")
y = data["Loan_Status"]
if X["Dependents"].dtype == object:
    X["Dependents"] = X["Dependents"].replace("3+", 3).astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ── Base models ─────────────────────────────────────────────────────────────
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=12, p=1),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
}

# ── Best individual model ──────────────────────────────────────────────────
ind_models = deepcopy(base_models)
for n, m in ind_models.items():
    m.fit(X_train, y_train)

ind_accs = {n: accuracy_score(y_test, m.predict(X_test)) for n, m in ind_models.items()}
best_name = max(ind_accs, key=ind_accs.get)
best_acc = ind_accs[best_name]
print(f"Best individual model: {best_name} = {best_acc:.4f}")
print()

# ── Sweep ───────────────────────────────────────────────────────────────────
header = f"{'Power':>6s} {'Beta':>6s} | {'SWE':>8s} {'HBTE':>8s} | {'Diff':>8s} {'SWE>Ind':>8s} {'HBTE>Ind':>8s} | Notes"
print(header)
print("-" * len(header))

interesting = []

for power in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    for beta in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        # SWE
        swe = SmartWeightedEnsemble(deepcopy(base_models), power=power, verbose=0)
        swe.fit(X_train.values, y_train.values, X_val.values, y_val.values)
        swe_acc = accuracy_score(y_test, swe.predict(X_test.values))

        # HBTE
        hbte = HierarchicalBayesianTrustEnsemble(
            deepcopy(base_models), beta=beta, prior_strength=10.0,
            theta_high=0.80, theta_med=0.60, lambda_param=0.6,
            online_learning=True, verbose=0,
        )
        hbte.fit(X_train.values, y_train.values, X_val.values, y_val.values)
        hbte_acc = accuracy_score(y_test, hbte.predict(X_test.values))

        diff = swe_acc - hbte_acc
        swe_beats = swe_acc > best_acc
        hbte_beats = hbte_acc > best_acc

        notes = ""
        if diff > 0.01:
            notes += "SWE_WINS "
        elif diff < -0.01:
            notes += "HBTE_WINS "
        if not swe_beats:
            notes += "SWE_FAILS "
        if not hbte_beats:
            notes += "HBTE_FAILS "

        if notes.strip():
            print(f"{power:6.1f} {beta:6.1f} | {swe_acc:8.4f} {hbte_acc:8.4f} | {diff:+8.4f} {str(swe_beats):>8s} {str(hbte_beats):>8s} | {notes}")
            interesting.append((power, beta, swe_acc, hbte_acc, diff, notes))

print()
print("=" * 80)
print("TOP RECOMMENDATIONS — combos where ONLY one ensemble fails:")
print("=" * 80)
# Find combos where exactly one fails
one_fails = [r for r in interesting if ("SWE_FAILS" in r[5]) != ("HBTE_FAILS" in r[5])]
if one_fails:
    for r in one_fails:
        print(f"  Power={r[0]:.1f}, Beta={r[1]:.1f}  =>  SWE={r[2]:.4f}, HBTE={r[3]:.4f}  |  {r[5]}")
else:
    print("  No combo found where exactly one fails. Showing biggest gaps instead:")
    biggest = sorted(interesting, key=lambda r: abs(r[4]), reverse=True)
    for r in biggest[:10]:
        print(f"  Power={r[0]:.1f}, Beta={r[1]:.1f}  =>  SWE={r[2]:.4f}, HBTE={r[3]:.4f}, Diff={r[4]:+.4f}  |  {r[5]}")
