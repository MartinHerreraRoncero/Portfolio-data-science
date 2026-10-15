"""
Módulo de Scorecard Regulatorio Basilea con Weight-of-Evidence (WoE) y Optimal Binning.
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500 y Mercado EE.UU.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from optbinning import BinningProcess, Scorecard
from pathlib import Path


class CreditScorecardWOE(BaseEstimator, ClassifierMixin):
    """
    Scorecard de Crédito y Riesgo de Insolvencia basado en Optimal Binning y WoE.
    Cumple con los estándares de Basilea II/III e IFRS 9 para modelos de concesión e interpretabilidad.
    """
    def __init__(self, variable_names=None, categorical_variables=None, scaling_min=300, scaling_max=850, C=0.1, random_state=42):
        self.variable_names = variable_names
        self.categorical_variables = categorical_variables or []
        self.scaling_min = scaling_min
        self.scaling_max = scaling_max
        self.C = C
        self.random_state = random_state
        self.scorecard = None
        self.binning_process = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y_arr = np.asarray(y).astype(int)

        if self.variable_names is None:
            self.variable_names = list(X_df.columns)

        cat_vars = [c for c in self.categorical_variables if c in self.variable_names]

        self.binning_process = BinningProcess(
            variable_names=self.variable_names,
            categorical_variables=cat_vars,
            min_prebin_size=0.03,
            min_bin_size=0.05,
            max_n_bins=6
        )

        estimator = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            max_iter=300,
            random_state=self.random_state
        )

        self.scorecard = Scorecard(
            binning_process=self.binning_process,
            estimator=estimator,
            scaling_method="min_max",
            scaling_method_params={"min": self.scaling_min, "max": self.scaling_max}
        )

        self.scorecard.fit(X_df, y_arr)
        return self

    def predict_proba(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.scorecard.predict_proba(X_df)

    def predict_proba_positive(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        probs = self.scorecard.predict_proba(X_df)
        return probs[:, 1] if len(probs.shape) == 2 else probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba_positive(X)
        return (probs >= threshold).astype(int)

    def score(self, X):
        """Calcula el puntaje de crédito (Score 300-850)."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.scorecard.score(X_df)

    def table(self, style="detailed"):
        """Devuelve la tabla de scorecard con bins, WoE, IV y Puntos."""
        return self.scorecard.table(style=style)

    def information_value_summary(self):
        """Resumen de Information Value (IV) por variable."""
        table = self.table(style="detailed")
        iv_df = table.groupby('Variable')['IV'].max().reset_index().sort_values('IV', ascending=False)
        return iv_df


if __name__ == "__main__":
    print("[Scorecard WOE] Probando Scorecard regulatorio...")
    np.random.seed(42)
    N = 1000
    df_dummy = pd.DataFrame({
        'working_capital': np.random.randn(N),
        'net_income': np.random.randn(N),
        'inflation': np.random.uniform(0, 1, N),
        'distance_to_default': np.random.exponential(2.0, N)
    })
    y_dummy = ((df_dummy['inflation'] > 0.6) | (df_dummy['working_capital'] < -1.0)).astype(int)

    sc = CreditScorecardWOE()
    sc.fit(df_dummy, y_dummy)
    probs = sc.predict_proba_positive(df_dummy)
    scores = sc.score(df_dummy)
    print(f"[Scorecard WOE] Entrenado con exito. Probs sample: {probs[:3]}, Scores sample: {scores[:3]}")
    print("\n[Scorecard WOE] Information Value Summary:")
    print(sc.information_value_summary().to_string())
