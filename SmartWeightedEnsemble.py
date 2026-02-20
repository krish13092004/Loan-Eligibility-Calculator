"""
Smart Weighted Ensemble (SWE) Algorithm
========================================

A simplified novel ensemble learning algorithm that combines 8 classifiers
using adaptive performance-based weighting and confidence scoring.

Author: [Your Name]
Date: December 22, 2025
For: Final Year Major Project - Loan Eligibility Prediction

This is a SIMPLIFIED version that's easier to understand and explain!
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class SmartWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Smart Weighted Ensemble (SWE) - A Novel Simplified Algorithm
    
    This algorithm combines K base classifiers using:
    1. Performance-based dynamic weighting (α^power formula)
    2. Prediction confidence scoring
    3. Model agreement analysis
    
    Parameters:
    -----------
    models : dict
        Dictionary of {model_name: model_instance}
    
    power : float, default=2.0
        Exponent for weight calculation (higher = more weight to best models)
    
    confidence_threshold : float, default=0.75
        Threshold for high-confidence predictions (0-1)
    
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress)
    
    Key Features:
    -------------
    - Simple to understand and implement
    - Weights based on validation accuracy
    - Provides confidence scores for predictions
    - Shows which models agree/disagree
    """
    
    def __init__(self, models, power=2.0, confidence_threshold=0.75, verbose=1):
        self.models = models
        self.power = power
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train all models and calculate weights based on accuracy.
        
        Weight Formula:
        ---------------
        weight_k = (accuracy_k ^ power) / Σ(accuracy_j ^ power)
        
        This gives higher weight to better-performing models.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("SWE currently only supports binary classification")
        
        if self.verbose:
            print("="*70)
            print("SMART WEIGHTED ENSEMBLE (SWE) - SIMPLIFIED NOVEL ALGORITHM")
            print("="*70)
            print(f"\nTraining {len(self.models)} base models...\n")
        
        # Train models and calculate accuracies
        accuracies = {}
        
        for name, model in self.models.items():
            if self.verbose:
                print(f"  > {name}...", end="")
            
            model.fit(X, y)
            
            # Get validation accuracy
            if X_val is not None and y_val is not None:
                acc = model.score(X_val, y_val)
            else:
                acc = model.score(X, y)
            
            accuracies[name] = acc
            
            if self.verbose:
                print(f" Accuracy: {acc:.4f}")
        
        # Calculate weights using power formula
        if self.verbose:
            print(f"\nCalculating weights (power = {self.power})...\n")
        
        powered_acc = {name: acc ** self.power for name, acc in accuracies.items()}
        total = sum(powered_acc.values())
        
        self.weights_ = {name: pa / total for name, pa in powered_acc.items()}
        self.accuracies_ = accuracies
        
        # Show weights
        if self.verbose:
            print("Model Weights:")
            for name in sorted(self.weights_.items(), key=lambda x: x[1], reverse=True):
                weight = self.weights_[name[0]]
                bar = '█' * int(weight * 50)
                print(f"  {name[0]:25s}: {weight:.4f} {bar}")
        
        # Find best model
        best = max(accuracies.items(), key=lambda x: x[1])
        
        if self.verbose:
            print(f"\n✓ Best model: {best[0]} ({best[1]:.4f})")
            print("="*70 + "\n")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probabilities using weighted combination.
        
        Formula:
        --------
        P(class=1|x) = Σ(weight_k × P_k(class=1|x)) / Σ(weight_k)
        """
        check_is_fitted(self, ['weights_'])
        X = check_array(X)
        
        n_samples = X.shape[0]
        weighted_probs = np.zeros((n_samples, len(self.classes_)))
        
        # Weighted sum of probabilities
        total_weight = sum(self.weights_.values())
        
        for name, model in self.models.items():
            proba = model.predict_proba(X)
            weight = self.weights_[name]
            weighted_probs += weight * proba
        
        # Normalize
        return weighted_probs / total_weight
    
    def predict(self, X, return_confidence=False):
        """
        Make predictions with optional confidence scores.
        
        Confidence Score:
        -----------------
        confidence = max(probability) × model_agreement
        
        Where model_agreement = fraction of models agreeing with prediction
        """
        proba = self.predict_proba(X)
        predictions = self.classes_[np.argmax(proba, axis=1)]
        
        if return_confidence:
            # Calculate confidence
            max_proba = np.max(proba, axis=1)
            
            # Get model agreement
            agreements = np.zeros(len(predictions))
            for name, model in self.models.items():
                model_preds = model.predict(X)
                agreements += (model_preds == predictions)
            
            agreement_ratio = agreements / len(self.models)
            
            # Combined confidence: probability × agreement
            confidence = max_proba * agreement_ratio
            
            return predictions, confidence
        
        return predictions
    
    def explain_prediction(self, X, idx=0):
        """
        Explain how the ensemble made a prediction for a sample.
        
        Returns:
        --------
        Dictionary with:
        - final_prediction: Ensemble decision
        - confidence: Confidence score (0-1)
        - individual_votes: Each model's prediction
        - is_high_confidence: Whether confidence exceeds threshold
        """
        check_is_fitted(self, ['weights_'])
        X = check_array(X)
        
        x = X[idx:idx+1]
        
        # Ensemble prediction
        pred, conf = self.predict(x, return_confidence=True)
        proba = self.predict_proba(x)[0]
        
        # Individual model votes
        votes = {}
        for name, model in self.models.items():
            p = model.predict(x)[0]
            prob = model.predict_proba(x)[0]
            votes[name] = {
                'prediction': 'Approved' if p == 1 else 'Rejected',
                'probability': float(prob[1]),
                'weight': float(self.weights_[name])
            }
        
        return {
            'final_prediction': 'Approved' if pred[0] == 1 else 'Rejected',
            'confidence': float(conf[0]),
            'probability_approved': float(proba[1]),
            'probability_rejected': float(proba[0]),
            'is_high_confidence': conf[0] >= self.confidence_threshold,
            'model_votes': votes,
            'num_models_agree': sum(1 for v in votes.values() 
                                   if v['prediction'] == ('Approved' if pred[0] == 1 else 'Rejected'))
        }
    
    def get_model_rankings(self):
        """
        Get models ranked by their weights.
        
        Returns list of (model_name, weight, accuracy) tuples.
        """
        check_is_fitted(self, ['weights_'])
        
        rankings = []
        for name in self.models.keys():
            rankings.append((
                name,
                self.weights_[name],
                self.accuracies_[name]
            ))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
