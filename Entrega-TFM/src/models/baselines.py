"""
Módulo de Modelos Base Estocásticos: Regresión Logística Regularizada.
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, average_precision_score

class BaselineLogisticModel:
    """Modelo baseline de Regresión Logística con regularización ElasticNet / L2 y sintonización de class_weight."""
    def __init__(self, penalty='l2', C=1.0, class_weight='balanced', random_state=42):
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            class_weight=class_weight,
            solver='lbfgs' if penalty == 'l2' else 'saga',
            max_iter=5000,
            random_state=random_state
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test, threshold=0.5):
        probs = self.predict_proba(X_test)
        preds = (probs >= threshold).astype(int)
        
        auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds)
        
        return {
            'roc_auc': float(auc),
            'pr_auc': float(pr_auc),
            'recall': float(recall),
            'precision': float(precision),
            'f1_score': float(f1),
            'probabilities': probs,
            'predictions': preds
        }


class EqualWeightVotingClassifier:
    """
    Ensemble de Votacion Suave (Soft Voting) con ponderacion estricta del 25%
    para cada uno de los 4 modelos lideres del benchmark:
    LightGBM, Random Forest, CatBoost y XGBoost.
    """
    def __init__(self, lgb_model=None, rf_model=None, cb_model=None, xgb_model=None, threshold=0.156):
        self.lgb = lgb_model
        self.rf = rf_model
        self.cb = cb_model
        self.xgb = xgb_model
        self.weights = [0.25, 0.25, 0.25, 0.25]
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p_lgb = self.lgb.predict_proba(X)
        if p_lgb.ndim > 1 and p_lgb.shape[1] > 1:
            p_lgb = p_lgb[:, 1]
        p_rf = self.rf.predict_proba(X)
        if p_rf.ndim > 1 and p_rf.shape[1] > 1:
            p_rf = p_rf[:, 1]
        p_cb = self.cb.predict_proba(X)
        if p_cb.ndim > 1 and p_cb.shape[1] > 1:
            p_cb = p_cb[:, 1]
        p_xgb = self.xgb.predict_proba(X)
        if p_xgb.ndim > 1 and p_xgb.shape[1] > 1:
            p_xgb = p_xgb[:, 1]
        p1 = 0.25 * p_lgb + 0.25 * p_rf + 0.25 * p_cb + 0.25 * p_xgb
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

