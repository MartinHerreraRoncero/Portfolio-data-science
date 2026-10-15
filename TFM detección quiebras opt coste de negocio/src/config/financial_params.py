"""
Modulo de Parametros Financieros y Umbrales Optimos de Decision de Credito (IFRS 9 / Basilea III).
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class CreditPortfolioConfig:
    ead_usd: float = 10_000_000.0       # Exposicion en el Incumplimiento por prestamo ($10M USD)
    nim_pct: float = 0.025              # Margen de Interes Neto anual (2.5%)
    lgd_pct: float = 0.600              # Perdida en caso de Incumplimiento (60.0%)
    
    @property
    def cost_fp_usd(self) -> float:
        """Coste de Falso Positivo: Margen comercial perdido por denegacion injustificada ($250,000 USD)."""
        return self.ead_usd * self.nim_pct

    @property
    def cost_fn_usd(self) -> float:
        """Coste de Falso Negativo: Perdida neta de capital por insolvencia no anticipada ($6,000,000 USD)."""
        return self.ead_usd * self.lgd_pct

    @property
    def profit_tn_usd(self) -> float:
        """Beneficio por Verdadero Negativo: Margen comercial devengado ($250,000 USD)."""
        return self.ead_usd * self.nim_pct

    @property
    def avoided_tp_usd(self) -> float:
        """Perdida evitada por Verdadero Positivo: Capital preservado ($6,000,000 USD)."""
        return self.ead_usd * self.lgd_pct

    @property
    def asymmetry_ratio(self) -> float:
        """Ratio de penalizacion asimetrica oficial (24.0x)."""
        return self.cost_fn_usd / self.cost_fp_usd

DEFAULT_PORTFOLIO_CONFIG = CreditPortfolioConfig()

# Catalogo Oficial de Umbrales Optimos de Decision (tau*) calibrados por minimizacion de coste asimetrico
OFFICIAL_THRESHOLDS_12M = {
    'XGBoost': 0.158,
    'LightGBM': 0.224,
    'CatBoost GPU': 0.038,
    'CatBoost': 0.038,
    'Random Forest': 0.182,
    'Logistic Regression': 0.170,
    'PyTorch MLP (Custom Loss)': 0.166,
    'Ensemble (PR-AUC Weighted Voting)': 0.220,
    'Scorecard_WOE_Basel': 0.110,
    'Altman_Merton_Baseline': 0.500
}

OFFICIAL_THRESHOLDS_24M = {
    'XGBoost': 0.184,
    'LightGBM': 0.256,
    'CatBoost GPU': 0.082,
    'CatBoost': 0.082,
    'Random Forest': 0.202,
    'Logistic Regression': 0.224,
    'PyTorch MLP (Custom Loss)': 0.186,
    'Ensemble (PR-AUC Weighted Voting)': 0.258
}

PRODUCTION_DEFAULT_THRESHOLD = 0.158
