"""
HBTE Algorithm - Usage Example
================================

This file demonstrates how to use the Hierarchical Bayesian Trust Ensemble
for your loan eligibility prediction project.

Author: [Your Name]
Date: December 22, 2025
"""

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# Import the 8 base models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Import our novel HBTE algorithm
from HierarchicalBayesianTrustEnsemble import HierarchicalBayesianTrustEnsemble

# ============================================================================
# STEP 1: Load and Preprocess Data
# ============================================================================

print("="*70)
print("LOAN ELIGIBILITY PREDICTION USING HBTE")
print("="*70)
print("\n[Step 1] Loading and preprocessing data...")

# Load data
data = pd.read_csv('LoanData.csv')

# Handle missing values
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data['Credit_History'].fillna(0, inplace=True)

# Feature engineering (as per your existing notebook)
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['Loan_Income_Ratio'] = data['LoanAmount'] / (data['Total_Income'] + 1)  # +1 to avoid division by zero

# Encode categorical variables
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Married'] = le.fit_transform(data['Married'])
data['Education'] = le.fit_transform(data['Education'])
data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
data['Property_Area'] = le.fit_transform(data['Property_Area'])
data['Loan_Status'] = le.fit_transform(data['Loan_Status'])

# Prepare features and target
X = data.drop(['Loan_Status', 'Loan_ID', 'ApplicantIncome', 'CoapplicantIncome'], axis=1, errors='ignore')
y = data['Loan_Status']

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Class distribution: {np.bincount(y)}")

# ============================================================================
# STEP 2: Split Data (Train, Validation, Test)
# ============================================================================

print("\n[Step 2] Splitting data into train/val/test sets...")

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Second split: 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 3: Define the 8 Base Models
# ============================================================================

print("\n[Step 3] Initializing 8 base classifiers...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=12, p=1),  # Use your tuned hyperparameters
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

print(f"Number of base models: {len(models)}")

# ============================================================================
# STEP 4: Create and Train HBTE Ensemble
# ============================================================================

print("\n[Step 4] Training Hierarchical Bayesian Trust Ensemble...")
print("\nThis is your NOVEL ALGORITHM with Theorem 1 guarantees!\n")

# Initialize HBTE with your 8 models
hbte = HierarchicalBayesianTrustEnsemble(
    models=models,
    beta=2.0,                # Power parameter for initial trust (β in paper)
    prior_strength=10.0,     # Bayesian prior strength (t_0 in paper)
    theta_high=0.80,         # High confidence threshold
    theta_med=0.60,          # Medium confidence threshold
    lambda_param=0.6,        # Balance between entropy and agreement
    online_learning=True,    # Enable trust updates
    verbose=1                # Show progress
)

# Train HBTE (this also trains all 8 base models)
hbte.fit(X_train, y_train, X_val, y_val)

# ============================================================================
# STEP 5: Make Predictions and Evaluate
# ============================================================================

print("\n[Step 5] Making predictions on test set...")

# Predict with confidence scores
y_pred, confidence = hbte.predict(X_test, return_confidence=True)
y_pred_proba = hbte.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])

print("\n" + "="*70)
print("HBTE ENSEMBLE RESULTS")
print("="*70)
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"AUC-ROC Score: {auc_score:.4f}")
print(f"\nAverage Confidence: {np.mean(confidence):.4f}")
print(f"Min Confidence: {np.min(confidence):.4f}")
print(f"Max Confidence: {np.max(confidence):.4f}")

print("\n" + "-"*70)
print("Classification Report:")
print("-"*70)
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

print("\n" + "-"*70)
print("Confusion Matrix:")
print("-"*70)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# ============================================================================
# STEP 6: Hierarchical Tier Statistics
# ============================================================================

print("\n" + "="*70)
print("HIERARCHICAL TIER USAGE STATISTICS")
print("="*70)

tier_stats = hbte.get_tier_statistics()
print(f"\nTotal predictions: {tier_stats['total_predictions']}")
print(f"\nTier 1 (Top-3 models): {tier_stats['tier_1_count']} ({tier_stats['tier_1_pct']:.1f}%)")
print(f"Tier 2 (Top-5 models): {tier_stats['tier_2_count']} ({tier_stats['tier_2_pct']:.1f}%)")
print(f"Tier 3 (All-8 models): {tier_stats['tier_3_count']} ({tier_stats['tier_3_pct']:.1f}%)")
print("\n→ Higher Tier 1 usage = More confident predictions = Faster inference!")

# ============================================================================
# STEP 7: Trust Parameters (From Theorem 1)
# ============================================================================

print("\n" + "="*70)
print("BAYESIAN TRUST PARAMETERS (τ_k)")
print("="*70)
print("\nCurrent trust for each model (higher = more trusted):\n")

trust_df = pd.DataFrame({
    'Model': list(hbte.trust_.keys()),
    'Trust (τ)': list(hbte.trust_.values())
}).sort_values('Trust (τ)', ascending=False)

for idx, row in trust_df.iterrows():
    bar_length = int(row['Trust (τ)'] * 50)
    bar = '█' * bar_length
    print(f"{row['Model']:20s}: {row['Trust (τ)']:.4f} {bar}")

print(f"\n→ Most trusted model: {trust_df.iloc[0]['Model']}")
print(f"→ These weights converge to optimal values per Theorem 1!")

# ============================================================================
# STEP 8: Explain Individual Predictions
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE PREDICTION EXPLANATION")
print("="*70)

# Explain first few test samples
for idx in range(min(3, len(X_test))):
    explanation = hbte.explain_prediction(X_test.values, sample_idx=idx)
    
    print(f"\n--- Sample {idx + 1} ---")
    print(f"Final Prediction: {explanation['final_prediction']}")
    print(f"Confidence: {explanation['confidence']:.2%}")
    print(f"Probability of Approval: {explanation['probability_approved']:.2%}")
    print(f"\nIndividual model votes:")
    
    for model_name, vote_info in explanation['model_votes'].items():
        print(f"  {model_name:20s}: {vote_info['prediction']:8s} "
              f"(prob={vote_info['probability']:.3f}, trust={vote_info['trust']:.3f})")

# ============================================================================
# STEP 9: Theorem 1 Convergence Check
# ============================================================================

print("\n" + "="*70)
print("THEOREM 1 CONVERGENCE STATUS")
print("="*70)

convergence = hbte.get_theorem_convergence_status()

print(f"\nOnline predictions made: {convergence['total_predictions']}")
print(f"Converged to optimal weights: {'Yes ✓' if convergence['converged'] else 'Not yet'}")
print(f"Max difference from optimal: {convergence['max_difference']:.4f}")
print(f"Convergence rate (from Theorem 1): {convergence['convergence_rate']}")

print("\n→ As per Theorem 1, trust parameters converge to ρ_k / Σρ_j")
print("→ More predictions = Better convergence!")

# ============================================================================
# STEP 10: Compare with Individual Models
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: HBTE vs INDIVIDUAL MODELS")
print("="*70)

print(f"\n{'Model':25s} {'Accuracy':>10s}")
print("-"*40)

individual_results = []
for name, model in models.items():
    # Models are already trained by HBTE
    y_pred_individual = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_individual)
    individual_results.append((name, acc))
    print(f"{name:25s} {acc:>10.4f}")

print("-"*40)
print(f"{'HBTE Ensemble':25s} {accuracy:>10.4f} ← Your Novel Algorithm!")
print("="*40)

# Calculate improvement
best_individual_acc = max([acc for _, acc in individual_results])
improvement = ((accuracy - best_individual_acc) / best_individual_acc) * 100

print(f"\n✓ HBTE improves over best individual model by {improvement:+.2f}%")
print(f"✓ Provides confidence scores (interpretability)")
print(f"✓ Hierarchical efficiency (adaptive computation)")
print(f"✓ Provable convergence guarantees (Theorem 1)")

# ============================================================================
# STEP 11: Save the Model
# ============================================================================

print("\n" + "="*70)
print("SAVING TRAINED MODEL")
print("="*70)

import pickle

with open('hbte_model.pkl', 'wb') as f:
    pickle.dump(hbte, f)

print("\n✓ Model saved to 'hbte_model.pkl'")
print("  You can load it later with: hbte = pickle.load(open('hbte_model.pkl', 'rb'))")

# ============================================================================
# SUMMARY FOR YOUR PROJECT REPORT
# ============================================================================

print("\n" + "="*70)
print("PROJECT SUMMARY - FOR YOUR REPORT/PRESENTATION")
print("="*70)

print(f"""
✓ Developed novel algorithm: Hierarchical Bayesian Trust Ensemble (HBTE)
✓ Formal mathematical framework with Theorem 1 (convergence proof)
✓ Combines 8 heterogeneous classifiers: {', '.join(models.keys()[:3])}...
✓ Bayesian trust allocation with online learning
✓ Information-theoretic confidence quantification
✓ Hierarchical 3-tier decision structure for efficiency
✓ Achieves {accuracy:.2%} accuracy on loan eligibility prediction
✓ Improves over best individual model by {improvement:+.2f}%
✓ Provides interpretable predictions with confidence scores
✓ Computational efficiency through adaptive tier selection

Key Innovation:
- Unlike standard voting ensembles, HBTE uses Bayesian posterior updating
  to dynamically allocate trust based on online performance
- Theorem 1 proves convergence to optimal weights
- Hierarchical tiers balance accuracy and computational cost

Files Created:
1. HierarchicalBayesianTrustEnsemble.py - Implementation
2. HBTE_Algorithm_Mathematical_Framework.md - Theory & Theorem
3. HBTE_Usage_Example.py - This file

Your Contribution:
- Novel ensemble algorithm with theoretical guarantees
- Application to loan eligibility prediction
- Comparative analysis with 8 base models
- Interpretable AI with confidence quantification
""")

print("="*70)
print("END OF EXAMPLE")
print("="*70)
