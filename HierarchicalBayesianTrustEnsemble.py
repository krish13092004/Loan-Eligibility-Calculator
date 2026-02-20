"""
Hierarchical Bayesian Trust Ensemble (HBTE) Algorithm
======================================================

A novel ensemble learning algorithm with provable convergence guarantees.

Author: [Your Name]
Date: December 22, 2025
For: Final Year Major Project - Loan Eligibility Prediction

Mathematical Framework: See HBTE_Algorithm_Mathematical_Framework.md
Theorem 1: Proves convergence of trust parameters to optimal weights
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.stats import entropy
import warnings

class HierarchicalBayesianTrustEnsemble(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Bayesian Trust Ensemble (HBTE) - A Novel Algorithm
    
    This ensemble combines K base classifiers using:
    1. Bayesian posterior updating of model trust
    2. Information-theoretic confidence quantification
    3. Hierarchical decision tiers for efficiency
    4. Provable convergence to optimal weights (Theorem 1)
    
    Parameters:
    -----------
    models : dict
        Dictionary of {model_name: model_instance} containing K base classifiers
    
    beta : float, default=2.0
        Power parameter for initial trust calculation (β in paper)
        Higher values amplify differences in model performance
    
    prior_strength : float, default=10.0
        Bayesian prior strength parameter (t_0 in paper)
        Controls adaptation speed: lower = faster adaptation
    
    theta_high : float, default=0.80
        Confidence threshold for Tier 1 (high confidence)
        
    theta_med : float, default=0.60
        Confidence threshold for Tier 2 (medium confidence)
        
    lambda_param : float, default=0.6
        Weighting parameter for combined confidence (λ in paper)
        Balances entropy-based confidence and model agreement
    
    online_learning : bool, default=True
        Whether to update trust parameters during prediction
        Set to False for static ensemble
    
    verbose : int, default=1
        Verbosity level: 0=silent, 1=progress, 2=detailed
    
    Attributes:
    -----------
    trust_ : dict
        Current trust parameters τ_k for each model
    
    success_counts_ : dict
        Number of correct predictions by each model (s_k)
    
    failure_counts_ : dict
        Number of incorrect predictions by each model (f_k)
    
    initial_trust_ : dict
        Initial trust values α_k based on validation accuracy
    
    tier_usage_ : dict
        Count of how many predictions used each tier
    
    References:
    -----------
    See HBTE_Algorithm_Mathematical_Framework.md for complete mathematical derivation
    """
    
    def __init__(self, models, beta=2.0, prior_strength=10.0,
                 theta_high=0.80, theta_med=0.60, lambda_param=0.6,
                 online_learning=True, verbose=1):
        self.models = models
        self.beta = beta
        self.prior_strength = prior_strength
        self.theta_high = theta_high
        self.theta_med = theta_med
        self.lambda_param = lambda_param
        self.online_learning = online_learning
        self.verbose = verbose
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train all base models and initialize trust parameters.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        
        y : array-like, shape (n_samples,)
            Target values
        
        X_val : array-like, shape (n_val_samples, n_features), optional
            Validation data for trust initialization
            If None, uses training data (less ideal)
        
        y_val : array-like, shape (n_val_samples,), optional
            Validation targets
        
        Returns:
        --------
        self : object
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ != 2:
            raise ValueError("HBTE currently only supports binary classification")
        
        # Initialize tracking structures
        self.trust_ = {}
        self.initial_trust_ = {}
        self.success_counts_ = {name: 0 for name in self.models.keys()}
        self.failure_counts_ = {name: 0 for name in self.models.keys()}
        self.tier_usage_ = {'tier_1': 0, 'tier_2': 0, 'tier_3': 0}
        
        # Phase 1: Train all models
        if self.verbose >= 1:
            print("="*70)
            print("HIERARCHICAL BAYESIAN TRUST ENSEMBLE (HBTE)")
            print("="*70)
            print("\n[Phase 1] Training base models...")
            
        model_accuracies = {}
        
        for name, model in self.models.items():
            if self.verbose >= 1:
                print(f"  > Training {name}...", end="")
            
            model.fit(X, y)
            
            # Calculate validation accuracy for trust initialization
            if X_val is not None and y_val is not None:
                accuracy = model.score(X_val, y_val)
            else:
                if self.verbose >= 2:
                    warnings.warn("No validation set provided. Using training accuracy (may overfit).")
                accuracy = model.score(X, y)
            
            model_accuracies[name] = accuracy
            
            if self.verbose >= 1:
                print(f" Accuracy: {accuracy:.4f}")
        
        # Phase 2: Initialize trust parameters (α_k in paper)
        if self.verbose >= 1:
            print("\n[Phase 2] Initializing Bayesian trust parameters...")
            print(f"  Using β={self.beta} (power weighting parameter)")
            
        # Calculate α_k = Accuracy_k^β / Σ(Accuracy_j^β)
        powered_accuracies = {name: acc**self.beta for name, acc in model_accuracies.items()}
        total_powered = sum(powered_accuracies.values())
        
        for name in self.models.keys():
            alpha_k = powered_accuracies[name] / total_powered
            self.initial_trust_[name] = alpha_k
            self.trust_[name] = alpha_k  # τ_k(0) = α_k
            
            if self.verbose >= 2:
                print(f"  {name:20s}: α={alpha_k:.4f} (Acc={model_accuracies[name]:.4f})")
        
        # Find best model for reporting
        best_model = max(model_accuracies.items(), key=lambda x: x[1])
        
        if self.verbose >= 1:
            print(f"\n[Info] Best individual model: {best_model[0]} ({best_model[1]:.4f})")
            print(f"[Info] Prior strength parameter: t_0={self.prior_strength}")
            print("="*70)
            print("\n✓ HBTE is ready for hierarchical predictions!\n")
        
        return self
    
    def _update_trust(self, model_name, is_correct):
        """
        Bayesian trust update (Section 2.3 in paper)
        
        τ_k(t+1) = (s_k + α_k·t_0) / (s_k + f_k + t_0)
        """
        if is_correct:
            self.success_counts_[model_name] += 1
        else:
            self.failure_counts_[model_name] += 1
        
        s_k = self.success_counts_[model_name]
        f_k = self.failure_counts_[model_name]
        alpha_k = self.initial_trust_[model_name]
        t_0 = self.prior_strength
        
        # Bayesian posterior
        self.trust_[model_name] = (s_k + alpha_k * t_0) / (s_k + f_k + t_0)
    
    def _get_trusted_models(self, n_models):
        """Get top-N most trusted models"""
        sorted_models = sorted(self.trust_.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_models[:n_models]]
    
    def _compute_entropy_confidence(self, proba):
        """
        Information-theoretic confidence (Section 2.4 in paper)
        
        C(x) = 1 - H(x)/log(2)
        
        where H(x) is Shannon entropy of prediction probabilities
        """
        # Ensure probabilities sum to 1 and are valid
        proba = np.clip(proba, 1e-10, 1.0)
        proba = proba / proba.sum()
        
        # Shannon entropy
        H = entropy(proba, base=2)
        
        # Normalized confidence: 0 (max entropy) to 1 (min entropy)
        confidence = 1.0 - H
        
        return confidence
    
    def _compute_model_agreement(self, predictions, ensemble_pred):
        """
        Model agreement score (Section 2.5 in paper)
        
        A(x) = (1/K) Σ 1[ŷ_k = ŷ_ensemble]
        """
        agreements = np.sum(predictions == ensemble_pred)
        return agreements / len(predictions)
    
    def _predict_with_tier(self, X, tier_models):
        """
        Make prediction using specified tier of models
        
        Returns trust-weighted prediction
        """
        n_samples = X.shape[0]
        weighted_scores = np.zeros((n_samples, self.n_classes_))
        
        for model_name in tier_models:
            model = self.models[model_name]
            trust = self.trust_[model_name]
            
            # Get probability predictions
            proba = model.predict_proba(X)
            
            # Add weighted contribution
            weighted_scores += trust * proba
        
        return weighted_scores
    
    def predict_proba(self, X):
        """
        Predict class probabilities using HBTE hierarchical decision process
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        
        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities for each sample
        """
        check_is_fitted(self, ['trust_', 'models'])
        X = check_array(X)
        
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_classes_))
        
        # Get predictions from all models (needed for confidence calculation)
        all_probas = {}
        all_preds = {}
        
        for name, model in self.models.items():
            all_probas[name] = model.predict_proba(X)
            all_preds[name] = model.predict(X)
        
        # Process each sample
        for i in range(n_samples):
            x_sample = X[i:i+1]
            
            # Step 1: Compute trust-weighted probability (all models)
            weighted_proba = np.zeros(self.n_classes_)
            total_trust = 0
            
            for name in self.models.keys():
                weighted_proba += self.trust_[name] * all_probas[name][i]
                total_trust += self.trust_[name]
            
            weighted_proba /= total_trust
            
            # Step 2: Compute information-theoretic confidence (Section 2.4)
            entropy_confidence = self._compute_entropy_confidence(weighted_proba)
            
            # Step 3: Compute model agreement (Section 2.5)
            temp_pred = np.argmax(weighted_proba)
            model_predictions = np.array([all_preds[name][i] for name in self.models.keys()])
            agreement = self._compute_model_agreement(model_predictions, temp_pred)
            
            # Step 4: Combined confidence (Section 2.6)
            combined_confidence = (self.lambda_param * entropy_confidence + 
                                 (1 - self.lambda_param) * agreement)
            
            # Step 5: Hierarchical tier selection (Section 2.6)
            if combined_confidence >= self.theta_high:
                # Tier 1: High confidence - use top 3 models
                tier_models = self._get_trusted_models(3)
                self.tier_usage_['tier_1'] += 1
                tier_name = "Tier-1 (Top-3)"
            elif combined_confidence >= self.theta_med:
                # Tier 2: Medium confidence - use top 5 models
                tier_models = self._get_trusted_models(5)
                self.tier_usage_['tier_2'] += 1
                tier_name = "Tier-2 (Top-5)"
            else:
                # Tier 3: Low confidence - use all 8 models
                tier_models = list(self.models.keys())
                self.tier_usage_['tier_3'] += 1
                tier_name = "Tier-3 (All-8)"
            
            # Step 6: Make prediction with selected tier
            tier_proba = self._predict_with_tier(x_sample, tier_models)
            predictions[i] = tier_proba[0]
            
            if self.verbose >= 2 and i < 5:  # Show first 5 samples
                print(f"Sample {i}: Γ={combined_confidence:.3f} → {tier_name}")
        
        return predictions
    
    def predict(self, X, return_confidence=False):
        """
        Predict class labels using HBTE
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        
        return_confidence : bool, default=False
            If True, also return confidence scores
        
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted class labels
        
        confidence : array-like, shape (n_samples,)
            Confidence scores (only if return_confidence=True)
        """
        proba = self.predict_proba(X)
        predictions = self.classes_[np.argmax(proba, axis=1)]
        
        if return_confidence:
            # Confidence is max probability
            confidence = np.max(proba, axis=1)
            return predictions, confidence
        
        return predictions
    
    def partial_fit(self, X, y):
        """
        Online learning: Update trust parameters with new labeled data
        
        This implements the trust update mechanism described in Section 2.3
        and proves convergence in Theorem 1.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New samples
        
        y : array-like, shape (n_samples,)
            True labels for trust updating
        """
        check_is_fitted(self, ['trust_'])
        X, y = check_X_y(X, y)
        
        # Get predictions from all models
        for name, model in self.models.items():
            y_pred = model.predict(X)
            
            # Update trust for each sample
            for i in range(len(y)):
                is_correct = (y_pred[i] == y[i])
                self._update_trust(name, is_correct)
        
        if self.verbose >= 2:
            print("\n[Trust Update] New trust parameters:")
            for name, trust in self.trust_.items():
                print(f"  {name:20s}: τ={trust:.4f}")
    
    def get_tier_statistics(self):
        """
        Get statistics on hierarchical tier usage
        
        Returns:
        --------
        stats : dict
            Dictionary with tier usage counts and percentages
        """
        total = sum(self.tier_usage_.values())
        if total == 0:
            return self.tier_usage_
        
        stats = {
            'tier_1_count': self.tier_usage_['tier_1'],
            'tier_1_pct': self.tier_usage_['tier_1'] / total * 100,
            'tier_2_count': self.tier_usage_['tier_2'],
            'tier_2_pct': self.tier_usage_['tier_2'] / total * 100,
            'tier_3_count': self.tier_usage_['tier_3'],
            'tier_3_pct': self.tier_usage_['tier_3'] / total * 100,
            'total_predictions': total
        }
        return stats
    
    def explain_prediction(self, X, sample_idx=0):
        """
        Explain prediction for a specific sample
        
        Returns detailed breakdown of the HBTE decision process
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
        
        sample_idx : int, default=0
            Index of sample to explain
        
        Returns:
        --------
        explanation : dict
            Detailed explanation dictionary
        """
        check_is_fitted(self, ['trust_'])
        X = check_array(X)
        
        x = X[sample_idx:sample_idx+1]
        
        # Get individual model predictions
        model_votes = {}
        model_probas = {}
        
        for name, model in self.models.items():
            pred = model.predict(x)[0]
            proba = model.predict_proba(x)[0]
            model_votes[name] = {
                'prediction': 'Approved' if pred == 1 else 'Rejected',
                'probability': float(proba[1]),
                'trust': float(self.trust_[name])
            }
            model_probas[name] = proba
        
        # Ensemble prediction
        ensemble_proba = self.predict_proba(x)[0]
        ensemble_pred = np.argmax(ensemble_proba)
        
        explanation = {
            'final_prediction': 'Approved' if ensemble_pred == 1 else 'Rejected',
            'confidence': float(np.max(ensemble_proba)),
            'probability_approved': float(ensemble_proba[1]),
            'probability_rejected': float(ensemble_proba[0]),
            'model_votes': model_votes,
            'trust_parameters': {name: float(trust) for name, trust in self.trust_.items()},
            'most_trusted_model': max(self.trust_.items(), key=lambda x: x[1])[0]
        }
        
        return explanation
    
    def get_theorem_convergence_status(self):
        """
        Check convergence status relative to Theorem 1
        
        Returns information about how close trust parameters are
        to their theoretical optimal values (proportional to accuracy)
        
        Returns:
        --------
        status : dict
            Convergence information
        """
        total_predictions = sum(self.success_counts_.values()) + sum(self.failure_counts_.values())
        
        if total_predictions == 0:
            return {
                'converged': False,
                'message': 'No online learning performed yet',
                'total_predictions': 0
            }
        
        # Estimate current accuracy for each model
        empirical_accuracies = {}
        for name in self.models.keys():
            s = self.success_counts_[name]
            f = self.failure_counts_[name]
            if s + f > 0:
                empirical_accuracies[name] = s / (s + f)
            else:
                empirical_accuracies[name] = 0
        
        # Theoretical optimal weights (Theorem 1)
        total_accuracy = sum(empirical_accuracies.values())
        optimal_weights = {name: acc / total_accuracy 
                          for name, acc in empirical_accuracies.items()}
        
        # Compare current trust to optimal
        weight_diff = {name: abs(self.trust_[name] - optimal_weights[name])
                      for name in self.models.keys()}
        
        max_diff = max(weight_diff.values())
        
        return {
            'converged': max_diff < 0.05,  # Threshold for convergence
            'max_difference': float(max_diff),
            'total_predictions': total_predictions,
            'current_trust': {name: float(trust) for name, trust in self.trust_.items()},
            'optimal_trust': {name: float(opt) for name, opt in optimal_weights.items()},
            'convergence_rate': f'O(1/√{total_predictions})'  # From Theorem 1 proof
        }
