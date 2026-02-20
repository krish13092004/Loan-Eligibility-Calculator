import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score

class AdaptiveWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    A Novel Ensemble Algorithm: Adaptive Confidence-Weighted Ensemble (ACWE)
    
    This algorithm improves upon standard voting classifiers by:
    1. Dynamically calculating model weights based on validation performance (Power-weighted)
    2. Integrating prediction confidence into the decision process
    3. Providing interpretability through 'Confidence Scores'
    
    It combines 8 different base classifiers:
    - Logistic Regression (Linear)
    - KNN (Instance-based)
    - Decision Tree (Rule-based)
    - Random Forest (Bagging)
    - SVM (Kernel-based)
    - XGBoost (Gradient Boosting)
    - AdaBoost (Adaptive Boosting)
    - Naive Bayes (Probabilistic)
    """
    
    def __init__(self, models):
        """
        Parameters:
        -----------
        models : dict
            Dictionary of instantiated classifiers aka the 8 algorithms
            Format: {'ModelName': model_instance}
        """
        self.models = models
        self.weights_ = {}
        self.best_model_name_ = ""
        self.best_model_score_ = 0.0
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit all models and calculate adaptive weights.
        
        If X_val and y_val are provided, weights are optimized based on validation set.
        Otherwise, weights are based on training score (less optimal due to overfitting risk).
        """
        # Validations
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        print("Training Adaptive Weighted Ensemble...")
        
        for name, model in self.models.items():
            print(f"  > Fitting {name}...")
            model.fit(X, y)
            
            # Calculate Performance for Weighting
            if X_val is not None and y_val is not None:
                score = model.score(X_val, y_val)
            else:
                score = model.score(X, y) # Fallback to training score
            
            # Power Weighting Strategy: Accuracy^2
            # This disproportionately rewards high-performing models and penalizes weak ones
            self.weights_[name] = score ** 2
            
            # Track best model
            if score > self.best_model_score_:
                self.best_model_score_ = score
                self.best_model_name_ = name
                
            print(f"    - Accuracy: {score:.4f} | Calculated Weight: {self.weights_[name]:.4f}")
            
        print(f"\n[Ensmeble Ready] Best Individual Model: {self.best_model_name_}")
        return self
        
    def predict_proba(self, X):
        """
        Returns weighted probability estimates for the ensemble.
        """
        check_is_fitted(self, ['weights_'])
        X = check_array(X)
        
        # Initialize weighted sum of probabilities
        weighted_probas = np.zeros((X.shape[0], len(self.classes_)))
        total_weight = sum(self.weights_.values())
        
        for name, model in self.models.items():
            probas = model.predict_proba(X)
            weight = self.weights_[name]
            weighted_probas += probas * weight
            
        # Normalize
        return weighted_probas / total_weight
    
    def predict(self, X):
        """
        Predict class labels for X.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    def get_confidence_score(self, X):
        """
        Calculates a 'Confidence Score' for each prediction.
        
        Score is a combination of:
        1. Probability Strength: How strong is the weighted probability?
        2. Model Agreement: What % of models agree with the final decision?
        """
        # 1. Probability Strength
        probas = self.predict_proba(X)
        prob_strength = np.max(probas, axis=1)
        
        # 2. Model Agreement
        final_preds = self.predict(X)
        n_models = len(self.models)
        agreements = np.zeros(X.shape[0])
        
        for name, model in self.models.items():
            model_preds = model.predict(X)
            agreements += (model_preds == final_preds)
            
        agreement_strength = agreements / n_models
        
        # Combined Confidence Metric (Weighted Average)
        # We give slightly more importance to the weighted probability strength
        confidence = (0.6 * prob_strength) + (0.4 * agreement_strength)
        return confidence

    def explain_prediction(self, X, sample_index=0):
        """
        Returns a text explanation of why a specific sample was classified this way.
        """
        X_sample = X[sample_index].reshape(1, -1)
        pred = self.predict(X_sample)[0]
        conf = self.get_confidence_score(X_sample)[0]
        
        explanation = {
            "prediction": "Approved" if pred == 1 else "Rejected",
            "confidence": f"{conf:.2%}",
            "model_votes": {}
        }
        
        for name, model in self.models.items():
            p = model.predict(X_sample)[0]
            explanation["model_votes"][name] = "Approved" if p == 1 else "Rejected"
            
        return explanation
