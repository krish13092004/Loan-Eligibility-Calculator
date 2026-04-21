

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class SmartWeightedEnsemble(BaseEstimator, ClassifierMixin):
   
    def __init__(self, models, power=2.0, confidence_threshold=0.75, verbose=1):
        self.models = models
        self.power = power
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
    def fit(self, X, y, X_val=None, y_val=None):
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
        check_is_fitted(self, ['weights_'])
        
        rankings = []
        for name in self.models.keys():
            rankings.append((
                name,
                self.weights_[name],
                self.accuracies_[name]
            ))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
