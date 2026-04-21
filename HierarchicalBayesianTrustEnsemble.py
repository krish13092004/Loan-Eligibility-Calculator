"""
Hierarchical Bayesian Trust Ensemble (HBTE)
============================================
A Novel Ensemble Learning Algorithm with Mathematical Guarantees

Key Features:
  1. Bayesian Posterior Updating of model trust (Beta-Bernoulli conjugate prior)
  2. Information-Theoretic Confidence via Shannon entropy
  3. Hierarchical 3-Tier Decision Structure (fast → thorough)
  4. Formal Convergence Theorem (Theorem 1): trust → true accuracy as t → ∞

The 8 base classifiers:
  Logistic Regression, KNN, Decision Tree, Random Forest,
  SVM, XGBoost, AdaBoost, Naive Bayes

Author : Krish
Date   : February 2026
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score


class HierarchicalBayesianTrustEnsemble(BaseEstimator, ClassifierMixin):
    """
    HBTE — Hierarchical Bayesian Trust Ensemble

    Parameters
    ----------
    models : dict
        Dictionary of {name: sklearn_estimator} for the base classifiers.
    beta : float, default=2.0
        Power parameter for initial trust calculation.
        Higher β amplifies accuracy differences between models.
    prior_strength : float, default=10.0
        t₀ in the Bayesian update — controls how quickly trust adapts.
        Larger values = trust changes more slowly (more conservative).
    theta_high : float, default=0.80
        Tier-1 threshold. If combined confidence Γ(x) ≥ θ_high,
        use only the top-3 most trusted models (fastest decision).
    theta_med : float, default=0.60
        Tier-2 threshold. If θ_med ≤ Γ(x) < θ_high,
        use the top-5 most trusted models (balanced).
        Below θ_med → Tier-3, use all models (most thorough).
    lambda_param : float, default=0.6
        Weight given to information-theoretic confidence vs model agreement
        in the combined confidence score Γ(x).
        Γ(x) = λ·C(x) + (1-λ)·A(x)
    online_learning : bool, default=True
        If True, update trust parameters after each prediction batch
        when ground truth is available.
    verbose : int, default=1
        0 = silent, 1 = progress output, 2 = detailed debug output.

    Mathematical Framework
    ----------------------
    Initial Trust:
        α_k = Accuracy_k^β / Σ(Accuracy_j^β)

    Bayesian Trust Update (Beta-Bernoulli):
        τ_k(t+1) = (s_k(t) + α_k·t₀) / (s_k(t) + f_k(t) + t₀)
        where s_k = successes, f_k = failures

    Information-Theoretic Confidence:
        H(x) = -Σ p̂_c · log(p̂_c)          (Shannon entropy)
        C(x) = 1 - H(x) / log(2)            (normalised confidence)

    Model Agreement:
        A(x) = (# models agreeing with ensemble) / K

    Combined Confidence:
        Γ(x) = λ·C(x) + (1-λ)·A(x)

    Hierarchical Decision:
        Tier 1 (Γ ≥ θ_high): top-3 trusted models   → fastest
        Tier 2 (θ_med ≤ Γ < θ_high): top-5 models    → balanced
        Tier 3 (Γ < θ_med): all K models              → most robust

    Theorem 1 (Convergence):
        As t → ∞,  τ_k(t) →ᵃ·ˢ· ρ_k / Σ ρ_j
        where ρ_k is the true accuracy of model M_k.
    """

    def __init__(
        self,
        models,
        beta=2.0,
        prior_strength=10.0,
        theta_high=0.80,
        theta_med=0.60,
        lambda_param=0.6,
        online_learning=True,
        verbose=1,
    ):
        self.models = models
        self.beta = beta
        self.prior_strength = prior_strength
        self.theta_high = theta_high
        self.theta_med = theta_med
        self.lambda_param = lambda_param
        self.online_learning = online_learning
        self.verbose = verbose

    # ──────────────────────────────────────────────────────────────────────
    #  FIT
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train all base models and initialise Bayesian trust parameters.

        Parameters
        ----------
        X, y        : Training features and labels.
        X_val, y_val: Validation set used to compute initial trust (α_k).
                      If not provided, training accuracy is used (less ideal).
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.verbose >= 1:
            print("=" * 70)
            print("  HIERARCHICAL BAYESIAN TRUST ENSEMBLE (HBTE)")
            print("=" * 70)
            print(f"  Base models : {len(self.models)}")
            print(f"  Beta (β)    : {self.beta}")
            print(f"  Prior t₀    : {self.prior_strength}")
            print(f"  Thresholds  : θ_high={self.theta_high}, θ_med={self.theta_med}")
            print(f"  Lambda (λ)  : {self.lambda_param}")
            print("=" * 70)
            print()
            print("  Training base models...")

        # --- Step 1: Train each base model and record validation accuracy ---
        self.accuracies_ = {}
        for name, model in self.models.items():
            model.fit(X, y)
            if X_val is not None and y_val is not None:
                acc = accuracy_score(y_val, model.predict(X_val))
            else:
                acc = accuracy_score(y, model.predict(X))
            self.accuracies_[name] = acc

            if self.verbose >= 1:
                print(f"    > {name:<25s} Accuracy: {acc:.4f}")

        # --- Step 2: Compute initial trust α_k = acc^β / Σ(acc^β) ---
        acc_arr = np.array(list(self.accuracies_.values()))
        powered = acc_arr ** self.beta
        powered_sum = powered.sum()

        self.alpha_ = {}   # initial prior trust
        self.trust_ = {}   # current trust (updated online)
        for i, name in enumerate(self.models):
            self.alpha_[name] = powered[i] / powered_sum
            self.trust_[name] = self.alpha_[name]

        # --- Step 3: Initialise Bayesian counters ---
        self.successes_ = {name: 0.0 for name in self.models}
        self.failures_  = {name: 0.0 for name in self.models}
        self.n_updates_ = 0

        # --- Step 4: Compute model ranking by trust ---
        self._update_ranking()

        # --- Tier tracking statistics ---
        self.tier_counts_ = {1: 0, 2: 0, 3: 0}
        self.total_predictions_ = 0

        if self.verbose >= 1:
            print()
            print("  Initial Trust Parameters (τ):")
            for name in self.ranked_models_:
                bar = "█" * int(self.trust_[name] * 100)
                print(f"    {name:<25s}: τ = {self.trust_[name]:.4f}  {bar}")
            print()
            print(f"  ✓ HBTE ready  |  Best: {self.ranked_models_[0]}")
            print("=" * 70)

        return self

    # ──────────────────────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────────────
    def _update_ranking(self):
        """Sort models by current trust (descending)."""
        self.ranked_models_ = sorted(
            self.trust_, key=self.trust_.get, reverse=True
        )

    def _get_tier_models(self, tier):
        """Return list of model names for the given tier."""
        if tier == 1:
            return self.ranked_models_[:3]
        elif tier == 2:
            return self.ranked_models_[:5]
        else:
            return list(self.models.keys())

    def _compute_entropy_confidence(self, weighted_prob):
        """
        Information-theoretic confidence for a single sample.

        C(x) = 1 - H(x) / log(2)
        where H(x) = -Σ p̂_c · log(p̂_c)  (Shannon entropy)
        """
        # Clip to avoid log(0)
        p = np.clip(weighted_prob, 1e-15, 1.0 - 1e-15)
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(self.n_classes_)  # log(2) for binary
        if max_entropy == 0:
            return 1.0
        confidence = 1.0 - (entropy / max_entropy)
        return np.clip(confidence, 0.0, 1.0)

    def _compute_agreement(self, individual_preds, ensemble_pred):
        """
        Model agreement score: fraction of models agreeing with ensemble.
        A(x) = (# agreeing) / K
        """
        agree = sum(1 for p in individual_preds.values() if p == ensemble_pred)
        return agree / len(individual_preds)

    # ──────────────────────────────────────────────────────────────────────
    #  PREDICT
    # ──────────────────────────────────────────────────────────────────────
    def predict(self, X, return_confidence=False):
        """
        Predict class labels using the hierarchical tier system.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        return_confidence : bool
            If True, also return the confidence score for each prediction.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
        confidence  : ndarray of shape (n_samples,) — only if return_confidence=True
        """
        check_is_fitted(self, ["trust_"])
        X = check_array(X)
        n_samples = X.shape[0]

        predictions = np.zeros(n_samples, dtype=self.classes_.dtype)
        confidences = np.zeros(n_samples)

        for i in range(n_samples):
            x_i = X[i:i+1]

            # --- Collect predictions and probabilities from ALL models ---
            all_preds = {}
            all_probas = {}
            for name, model in self.models.items():
                all_preds[name] = model.predict(x_i)[0]
                all_probas[name] = model.predict_proba(x_i)[0]

            # --- Compute trust-weighted ensemble probability (using all models first) ---
            trust_sum = sum(self.trust_.values())
            weighted_prob = np.zeros(self.n_classes_)
            for name in self.models:
                weighted_prob += self.trust_[name] * all_probas[name]
            weighted_prob /= trust_sum

            # --- Preliminary ensemble prediction (for agreement calculation) ---
            prelim_pred = self.classes_[np.argmax(weighted_prob)]

            # --- Information-theoretic confidence C(x) ---
            info_conf = self._compute_entropy_confidence(weighted_prob)

            # --- Model agreement A(x) ---
            agreement = self._compute_agreement(all_preds, prelim_pred)

            # --- Combined confidence Γ(x) = λ·C(x) + (1-λ)·A(x) ---
            gamma = self.lambda_param * info_conf + (1 - self.lambda_param) * agreement

            # --- Hierarchical tier selection ---
            if gamma >= self.theta_high:
                tier = 1
            elif gamma >= self.theta_med:
                tier = 2
            else:
                tier = 3

            self.tier_counts_[tier] += 1
            self.total_predictions_ += 1

            # --- Final prediction using the selected tier's models ---
            tier_models = self._get_tier_models(tier)
            tier_trust_sum = sum(self.trust_[n] for n in tier_models)
            tier_weighted_prob = np.zeros(self.n_classes_)
            for name in tier_models:
                tier_weighted_prob += self.trust_[name] * all_probas[name]
            tier_weighted_prob /= tier_trust_sum

            predictions[i] = self.classes_[np.argmax(tier_weighted_prob)]
            confidences[i] = gamma

            if self.verbose >= 2:
                print(f"  Sample {i}: Tier-{tier}  Γ={gamma:.3f}  "
                      f"C={info_conf:.3f}  A={agreement:.3f}  "
                      f"→ {predictions[i]}")

        if return_confidence:
            return predictions, confidences
        return predictions

    # ──────────────────────────────────────────────────────────────────────
    #  PREDICT_PROBA
    # ──────────────────────────────────────────────────────────────────────
    def predict_proba(self, X):
        """
        Return trust-weighted class probabilities (using all models).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        probas : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ["trust_"])
        X = check_array(X)
        n_samples = X.shape[0]

        trust_sum = sum(self.trust_.values())
        probas = np.zeros((n_samples, self.n_classes_))

        for name, model in self.models.items():
            model_proba = model.predict_proba(X)
            probas += self.trust_[name] * model_proba

        probas /= trust_sum
        return probas

    # ──────────────────────────────────────────────────────────────────────
    #  BAYESIAN TRUST UPDATE (ONLINE LEARNING)
    # ──────────────────────────────────────────────────────────────────────
    def update_trust(self, X, y_true):
        """
        Update trust parameters using Bayesian posterior (Beta-Bernoulli).

        τ_k(t+1) = (s_k + α_k · t₀) / (s_k + f_k + t₀)

        Parameters
        ----------
        X      : Features of observed samples.
        y_true : True labels for those samples.
        """
        check_is_fitted(self, ["trust_"])
        X = check_array(X)

        for name, model in self.models.items():
            preds = model.predict(X)
            correct = (preds == y_true)
            self.successes_[name] += np.sum(correct)
            self.failures_[name]  += np.sum(~correct)

            # Bayesian posterior update
            s = self.successes_[name]
            f = self.failures_[name]
            t0 = self.prior_strength
            alpha = self.alpha_[name]

            self.trust_[name] = (s + alpha * t0) / (s + f + t0)

        self.n_updates_ += len(y_true)

        # Normalise trust so they sum to 1
        total = sum(self.trust_.values())
        if total > 0:
            for name in self.trust_:
                self.trust_[name] /= total

        # Re-rank models
        self._update_ranking()

        if self.verbose >= 1:
            print(f"\n  ⟳ Trust updated ({self.n_updates_} total observations)")
            for name in self.ranked_models_[:3]:
                print(f"    Top: {name:<25s} τ = {self.trust_[name]:.4f}")

    # ──────────────────────────────────────────────────────────────────────
    #  TIER STATISTICS
    # ──────────────────────────────────────────────────────────────────────
    def get_tier_statistics(self):
        """
        Return dictionary of tier usage statistics.

        Returns
        -------
        dict with keys: total_predictions, tier_1_count, tier_1_pct, etc.
        """
        total = self.total_predictions_
        if total == 0:
            return {"total_predictions": 0}

        return {
            "total_predictions": total,
            "tier_1_count": self.tier_counts_[1],
            "tier_2_count": self.tier_counts_[2],
            "tier_3_count": self.tier_counts_[3],
            "tier_1_pct": self.tier_counts_[1] / total * 100,
            "tier_2_pct": self.tier_counts_[2] / total * 100,
            "tier_3_pct": self.tier_counts_[3] / total * 100,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  EXPLAIN PREDICTION
    # ──────────────────────────────────────────────────────────────────────
    def explain_prediction(self, X, idx=0):
        """
        Provide a detailed, interpretable explanation for a single prediction.

        Parameters
        ----------
        X   : array-like of shape (n_samples, n_features)
        idx : int, index of the sample to explain.

        Returns
        -------
        dict with prediction details, confidence breakdown, model votes, tier used.
        """
        check_is_fitted(self, ["trust_"])
        X = check_array(X)
        x_i = X[idx:idx+1]

        # Collect all model outputs
        all_preds = {}
        all_probas = {}
        for name, model in self.models.items():
            all_preds[name] = model.predict(x_i)[0]
            all_probas[name] = model.predict_proba(x_i)[0]

        # Trust-weighted probability
        trust_sum = sum(self.trust_.values())
        weighted_prob = np.zeros(self.n_classes_)
        for name in self.models:
            weighted_prob += self.trust_[name] * all_probas[name]
        weighted_prob /= trust_sum

        final_pred = self.classes_[np.argmax(weighted_prob)]

        # Confidence components
        info_conf = self._compute_entropy_confidence(weighted_prob)
        agreement = self._compute_agreement(all_preds, final_pred)
        gamma = self.lambda_param * info_conf + (1 - self.lambda_param) * agreement

        # Tier
        if gamma >= self.theta_high:
            tier = 1
            tier_label = "Tier 1 (High Confidence — Top 3 models)"
        elif gamma >= self.theta_med:
            tier = 2
            tier_label = "Tier 2 (Medium Confidence — Top 5 models)"
        else:
            tier = 3
            tier_label = "Tier 3 (Low Confidence — All 8 models)"

        tier_models = self._get_tier_models(tier)

        # Build per-model breakdown
        model_details = {}
        for name in self.models:
            model_details[name] = {
                "prediction": "Approved" if all_preds[name] == 1 else "Rejected",
                "probability_approved": float(all_probas[name][1]) if self.n_classes_ > 1 else float(all_probas[name][0]),
                "trust": float(self.trust_[name]),
                "used_in_tier": name in tier_models,
            }

        return {
            "final_prediction": "Approved" if final_pred == 1 else "Rejected",
            "confidence_gamma": float(gamma),
            "information_confidence": float(info_conf),
            "model_agreement": float(agreement),
            "probability_approved": float(weighted_prob[1]) if self.n_classes_ > 1 else float(weighted_prob[0]),
            "tier_used": tier,
            "tier_label": tier_label,
            "models_used_count": len(tier_models),
            "num_models_agree": sum(1 for p in all_preds.values() if p == final_pred),
            "total_models": len(self.models),
            "model_details": model_details,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  STRING REPRESENTATION
    # ──────────────────────────────────────────────────────────────────────
    def __repr__(self):
        return (
            f"HierarchicalBayesianTrustEnsemble("
            f"models={len(self.models)}, β={self.beta}, "
            f"t₀={self.prior_strength}, "
            f"θ_high={self.theta_high}, θ_med={self.theta_med}, "
            f"λ={self.lambda_param})"
        )
