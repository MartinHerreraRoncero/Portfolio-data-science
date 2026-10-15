"""
Módulo de Regímenes Macroeconómicos y Arquetipos Corporativos mediante Modelos de Mezcla Gaussiana (GMM).
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.

Funcionalidades principales:
1. Selección óptima de hiperparámetros (K componentes in [2, 5], tipos de covarianza ['full', 'diag']) mediante Criterio de Información Bayesiano (BIC) y Akaike (AIC).
2. Segmentación de Regímenes Macroeconómicos (K=3):
   - Régimen 0: Expansión y Estabilidad Crediticia.
   - Régimen 1: Sobrecalentamiento / Endurecimiento Monetario.
   - Régimen 2: Estrés Crediticio / Recesión.
   - Generación de probabilidades a posteriori continuas:
     * macro_regime_expansion_prob
     * macro_regime_neutral_prob
     * macro_regime_crisis_prob
3. Segmentación de Arquetipos Financieros Corporativos (K=4) sobre solvencia y liquidez:
   - Características: tag_WorkingCapital, tag_NetIncomeLoss, tag_AssetsCurrent, tag_LiabilitiesCurrent.
   - Generación de probabilidades a posteriori continuas:
     * corp_archetype_prob_0, corp_archetype_prob_1, corp_archetype_prob_2, corp_archetype_prob_3
4. Exportación y persistencia de modelos entrenados (joblib/pickle), actualización del dataset Parquet máster y generación de informe JSON de auditoría.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MacroGMMRegimes")

# Rutas estándar del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "sec_dataset" / "v2_sec_financials_pivoted.parquet"
DEFAULT_CLEAN_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "sec_dataset" / "v2_sec_financials_pivoted_clean.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "data" / "processed" / "sec_dataset"

MACRO_FEATURES_DEFAULT = [
    "macro_interest_rate",
    "macro_yield_curve",
    "macro_real_gdp_growth_yoy",
    "macro_inflation",
    "macro_unemployment_rate"
]

CORP_FEATURES_DEFAULT = [
    "tag_WorkingCapital",
    "tag_NetIncomeLoss",
    "tag_AssetsCurrent",
    "tag_LiabilitiesCurrent"
]


class MacroGMMRegimeDetector:
    """
    Detector y Clasificador Probabilístico de Regímenes Macroeconómicos mediante Gaussian Mixture Models (GMM).
    """

    REGIME_NAMES = {
        0: "Expansión y Estabilidad Crediticia",
        1: "Sobrecalentamiento / Endurecimiento Monetario",
        2: "Estrés Crediticio / Recesión"
    }

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
        n_init: int = 10,
        macro_cols: Optional[List[str]] = None
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_init = n_init
        self.macro_cols = macro_cols or MACRO_FEATURES_DEFAULT

        self.scaler = StandardScaler()
        self.gmm: Optional[GaussianMixture] = None
        self.cluster_to_regime_map: Dict[int, int] = {}
        self.regime_to_cluster_map: Dict[int, int] = {}
        self.bic_table: Optional[pd.DataFrame] = None
        self.regime_means_: Optional[pd.DataFrame] = None
        self.regime_weights_: Optional[np.ndarray] = None

    def evaluate_bic(
        self,
        X_df: pd.DataFrame,
        k_range: range = range(2, 6),
        cov_types: List[str] = ["full", "diag"]
    ) -> pd.DataFrame:
        """
        Evalúa el Criterio de Información Bayesiano (BIC) y Criterio de Información de Akaike (AIC)
        para una malla de componentes K y estructuras de covarianza.
        """
        logger.info("[Macro GMM] Evaluando BIC/AIC para K in %s y cov_types in %s...", list(k_range), cov_types)
        
        # Extraer y normalizar variables macro
        X_raw = X_df[self.macro_cols].dropna().values
        scaler_temp = StandardScaler()
        X_scaled = scaler_temp.fit_transform(X_raw)

        results = []
        for cov in cov_types:
            for k in k_range:
                gmm_test = GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    random_state=self.random_state,
                    n_init=self.n_init
                )
                gmm_test.fit(X_scaled)
                bic_val = float(gmm_test.bic(X_scaled))
                aic_val = float(gmm_test.aic(X_scaled))
                log_lik = float(gmm_test.score(X_scaled) * len(X_scaled))
                results.append({
                    "covariance_type": cov,
                    "K": k,
                    "BIC": round(bic_val, 2),
                    "AIC": round(aic_val, 2),
                    "log_likelihood": round(log_lik, 2),
                    "converged": bool(gmm_test.converged_),
                    "n_iter": int(gmm_test.n_iter_)
                })

        self.bic_table = pd.DataFrame(results).sort_values("BIC", ascending=True).reset_index(drop=True)
        return self.bic_table

    def fit(self, df: pd.DataFrame) -> "MacroGMMRegimeDetector":
        """
        Ajusta el GMM sobre las variables macroeconómicas y alinea determinísticamente los clusters
        con los 3 regímenes económicos especificados.
        """
        logger.info("[Macro GMM] Ajustando GaussianMixture(K=%d, cov='%s')...", self.n_components, self.covariance_type)
        
        # Filtrar columnas macro requeridas
        missing_cols = [c for c in self.macro_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Variables macro faltantes en el DataFrame: {missing_cols}")

        X_raw = df[self.macro_cols].copy()
        # Imputar mediana si existiera algún NaN aislado
        if X_raw.isna().any().any():
            X_raw = X_raw.fillna(X_raw.median())

        X_scaled = self.scaler.fit_transform(X_raw.values)

        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=self.n_init
        )
        self.gmm.fit(X_scaled)

        if not self.gmm.converged_:
            logger.warning("[Macro GMM] Atención: El algoritmo EM no convergió completamente en %d iteraciones.", self.gmm.n_iter_)

        # Obtener centros en escala original
        raw_means = self.scaler.inverse_transform(self.gmm.means_)
        
        # Mapeo determinista basado en fundamentos económicos:
        # - Régimen 1 (Sobrecalentamiento / Endurecimiento): Mayor tipo de interés y mayor inflación / curva invertida
        # - Régimen 2 (Estrés Crediticio / Recesión): Mayor tasa de desempleo y menor crecimiento PIB YoY
        # - Régimen 0 (Expansión y Estabilidad): Restante (alto crecimiento PIB, baja inflación, desempleo moderado)
        
        # Índices de columnas
        rate_idx = self.macro_cols.index("macro_interest_rate")
        unemp_idx = self.macro_cols.index("macro_unemployment_rate")
        
        idx_tightening = int(np.argmax(raw_means[:, rate_idx]))
        idx_crisis = int(np.argmax(raw_means[:, unemp_idx]))
        
        # Si por alguna razón coincidieran, desempatar con GDP o inflación
        if idx_crisis == idx_tightening:
            gdp_idx = self.macro_cols.index("macro_real_gdp_growth_yoy")
            idx_crisis = int(np.argmin(raw_means[:, gdp_idx]))

        remaining = [i for i in range(self.n_components) if i not in [idx_tightening, idx_crisis]]
        idx_expansion = remaining[0] if remaining else 0

        self.regime_to_cluster_map = {
            0: idx_expansion,
            1: idx_tightening,
            2: idx_crisis
        }
        self.cluster_to_regime_map = {
            idx_expansion: 0,
            idx_tightening: 1,
            idx_crisis: 2
        }

        # Organizar tabla de centros según el régimen
        sorted_means = []
        sorted_weights = []
        for regime_id in range(3):
            orig_cluster = self.regime_to_cluster_map[regime_id]
            sorted_means.append(raw_means[orig_cluster])
            sorted_weights.append(self.gmm.weights_[orig_cluster])

        self.regime_means_ = pd.DataFrame(
            sorted_means,
            columns=self.macro_cols,
            index=[f"Régimen {r}: {self.REGIME_NAMES[r]}" for r in range(3)]
        )
        self.regime_weights_ = np.array(sorted_weights)

        logger.info("[Macro GMM] Modelo ajustado exitosamente. Pesos de mezcla ordenados: %s", np.round(self.regime_weights_, 4))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcula las probabilidades a posteriori continuas P(Regimen = r | X_macro)
        ordenadas por [P(Expansion), P(Neutral/Tightening), P(Crisis)].
        """
        if self.gmm is None:
            raise RuntimeError("El modelo MacroGMMRegimeDetector no ha sido entrenado. Llame a fit() primero.")

        X_raw = df[self.macro_cols].copy()
        if X_raw.isna().any().any():
            X_raw = X_raw.fillna(X_raw.median())

        X_scaled = self.scaler.transform(X_raw.values)
        raw_probs = self.gmm.predict_proba(X_scaled)  # shape: (N, 3)

        # Reordenar columnas a los regímenes [0, 1, 2]
        ordered_probs = np.zeros_like(raw_probs)
        for regime_id in range(3):
            orig_cluster = self.regime_to_cluster_map[regime_id]
            ordered_probs[:, regime_id] = raw_probs[:, orig_cluster]

        # Normalizar para garantizar suma estricta a 1.0 (evitar derivas numéricas)
        sum_p = ordered_probs.sum(axis=1, keepdims=True)
        sum_p[sum_p == 0] = 1.0
        ordered_probs = ordered_probs / sum_p

        return ordered_probs

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade las columnas de probabilidad y etiquetas de régimen al DataFrame.
        """
        probs = self.predict_proba(df)
        out_df = df.copy()

        out_df["macro_regime_expansion_prob"] = probs[:, 0].astype(np.float32)
        out_df["macro_regime_neutral_prob"] = probs[:, 1].astype(np.float32)
        out_df["macro_regime_crisis_prob"] = probs[:, 2].astype(np.float32)
        
        # Régimen discreto mayoritario
        discrete_regimes = np.argmax(probs, axis=1).astype(np.int32)
        out_df["macro_regime_label"] = discrete_regimes
        out_df["macro_regime_name"] = [self.REGIME_NAMES[r] for r in discrete_regimes]

        return out_df


class CorporateArchetypeGMMDetector:
    """
    Detector y Segmentador de Arquetipos Financieros Corporativos de Solvencia y Liquidez
    utilizando Gaussian Mixture Models (K=4).
    """

    ARCHETYPE_NAMES = {
        0: "Arquetipo 0: Micro-Cap con Estrés de Liquidez / Alta Vulnerabilidad",
        1: "Arquetipo 1: Mid-Cap con Operación y Liquidez Estable",
        2: "Arquetipo 2: Large-Cap con Alta Solvencia y Liquidez Robusta",
        3: "Arquetipo 3: Mega-Cap con Balance Fortaleza (Near-Zero Risk)"
    }

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "full",
        random_state: int = 42,
        n_init: int = 5,
        corp_cols: Optional[List[str]] = None
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_init = n_init
        self.corp_cols = corp_cols or CORP_FEATURES_DEFAULT

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.gmm: Optional[GaussianMixture] = None
        self.clip_lower_: Optional[np.ndarray] = None
        self.clip_upper_: Optional[np.ndarray] = None
        self.cluster_to_archetype_map: Dict[int, int] = {}
        self.archetype_to_cluster_map: Dict[int, int] = {}
        self.bic_table: Optional[pd.DataFrame] = None
        self.archetype_means_: Optional[pd.DataFrame] = None
        self.archetype_weights_: Optional[np.ndarray] = None

    def evaluate_bic(
        self,
        X_df: pd.DataFrame,
        k_range: range = range(2, 6),
        cov_types: List[str] = ["full", "diag"],
        sample_size: int = 50000
    ) -> pd.DataFrame:
        """
        Evalúa BIC y AIC sobre una muestra representativa o dataset completo.
        """
        logger.info("[Corp Archetypes GMM] Evaluando BIC/AIC para K in %s...", list(k_range))
        
        X_raw = X_df[self.corp_cols].copy()
        X_imp = SimpleImputer(strategy="median").fit_transform(X_raw)
        
        p1 = np.percentile(X_imp, 1, axis=0)
        p99 = np.percentile(X_imp, 99, axis=0)
        X_clip = np.clip(X_imp, p1, p99)
        X_scaled = StandardScaler().fit_transform(X_clip)

        if len(X_scaled) > sample_size:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X_scaled), size=sample_size, replace=False)
            X_eval = X_scaled[idx]
        else:
            X_eval = X_scaled

        results = []
        for cov in cov_types:
            for k in k_range:
                gmm_test = GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    random_state=self.random_state,
                    n_init=self.n_init
                )
                gmm_test.fit(X_eval)
                bic_val = float(gmm_test.bic(X_eval))
                aic_val = float(gmm_test.aic(X_eval))
                results.append({
                    "covariance_type": cov,
                    "K": k,
                    "BIC": round(bic_val, 2),
                    "AIC": round(aic_val, 2),
                    "converged": bool(gmm_test.converged_),
                    "n_iter": int(gmm_test.n_iter_)
                })

        self.bic_table = pd.DataFrame(results).sort_values("BIC", ascending=True).reset_index(drop=True)
        return self.bic_table

    def fit(self, df: pd.DataFrame, sample_size: Optional[int] = None) -> "CorporateArchetypeGMMDetector":
        """
        Ajusta el GMM sobre variables corporativas con winsorización robusta e imputación.
        Ordena los arquetipos monótonamente por solidez de balance (0: Más vulnerable -> 3: Mega fortaleza).
        """
        logger.info("[Corp Archetypes GMM] Ajustando GaussianMixture(K=%d, cov='%s')...", self.n_components, self.covariance_type)
        
        missing_cols = [c for c in self.corp_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Variables corporativas faltantes en el DataFrame: {missing_cols}")

        X_raw = df[self.corp_cols].copy()
        X_imp = self.imputer.fit_transform(X_raw)

        # Winsorización al percentil 1 y 99 para evitar distorsiones por outliers contables
        self.clip_lower_ = np.percentile(X_imp, 1, axis=0)
        self.clip_upper_ = np.percentile(X_imp, 99, axis=0)
        X_clip = np.clip(X_imp, self.clip_lower_, self.clip_upper_)

        X_scaled = self.scaler.fit_transform(X_clip)

        if sample_size and len(X_scaled) > sample_size:
            rng = np.random.RandomState(self.random_state)
            sample_idx = rng.choice(len(X_scaled), size=sample_size, replace=False)
            X_fit = X_scaled[sample_idx]
        else:
            X_fit = X_scaled

        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=self.n_init
        )
        self.gmm.fit(X_fit)

        # Obtener medias en escala original
        raw_means = self.scaler.inverse_transform(self.gmm.means_)

        # Ordenar arquetipos por tamaño de WorkingCapital / Total Liquidity (Activos Circulantes)
        # de menor a mayor (0: Más vulnerable / menor liquidez -> 3: Mayor fortaleza)
        wc_idx = self.corp_cols.index("tag_WorkingCapital")
        assets_curr_idx = self.corp_cols.index("tag_AssetsCurrent")
        
        # Métrica combinada de ranking de liquidez: WC + Activos Circulantes
        liquidity_rank_metric = raw_means[:, wc_idx] + raw_means[:, assets_curr_idx]
        sorted_clusters = np.argsort(liquidity_rank_metric)  # de menor a mayor

        self.archetype_to_cluster_map = {archetype_id: int(orig_c) for archetype_id, orig_c in enumerate(sorted_clusters)}
        self.cluster_to_archetype_map = {int(orig_c): archetype_id for archetype_id, orig_c in enumerate(sorted_clusters)}

        sorted_means = [raw_means[self.archetype_to_cluster_map[a]] for a in range(self.n_components)]
        sorted_weights = [self.gmm.weights_[self.archetype_to_cluster_map[a]] for a in range(self.n_components)]

        self.archetype_means_ = pd.DataFrame(
            sorted_means,
            columns=self.corp_cols,
            index=[f"Arquetipo {a}" for a in range(self.n_components)]
        )
        self.archetype_weights_ = np.array(sorted_weights)

        logger.info("[Corp Archetypes GMM] Ajuste completado. Pesos de mezcla ordenados: %s", np.round(self.archetype_weights_, 4))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcula las probabilidades a posteriori continuas P(Arquetipo = k | X_corp)
        para k in [0, 1, 2, 3].
        """
        if self.gmm is None:
            raise RuntimeError("El modelo CorporateArchetypeGMMDetector no ha sido entrenado.")

        X_raw = df[self.corp_cols].copy()
        X_imp = self.imputer.transform(X_raw)
        X_clip = np.clip(X_imp, self.clip_lower_, self.clip_upper_)
        X_scaled = self.scaler.transform(X_clip)

        raw_probs = self.gmm.predict_proba(X_scaled)  # shape: (N, 4)

        ordered_probs = np.zeros_like(raw_probs)
        for archetype_id in range(self.n_components):
            orig_cluster = self.archetype_to_cluster_map[archetype_id]
            ordered_probs[:, archetype_id] = raw_probs[:, orig_cluster]

        sum_p = ordered_probs.sum(axis=1, keepdims=True)
        sum_p[sum_p == 0] = 1.0
        ordered_probs = ordered_probs / sum_p

        return ordered_probs

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade las 4 columnas de probabilidad continua y etiqueta de arquetipo al DataFrame.
        """
        probs = self.predict_proba(df)
        out_df = df.copy()

        for k in range(self.n_components):
            out_df[f"corp_archetype_prob_{k}"] = probs[:, k].astype(np.float32)

        discrete_archetypes = np.argmax(probs, axis=1).astype(np.int32)
        out_df["corp_archetype_label"] = discrete_archetypes
        out_df["corp_archetype_name"] = [self.ARCHETYPE_NAMES.get(a, f"Arquetipo {a}") for a in discrete_archetypes]

        return out_df


def run_macro_corporate_gmm_pipeline(
    input_parquet_path: Path = DEFAULT_PARQUET_PATH,
    output_parquet_path: Optional[Path] = None,
    save_models: bool = True,
    save_report: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta el pipeline completo de Modelos GMM Macroeconómicos y Corporativos:
    1. Carga del dataset maestro.
    2. Evaluación BIC/AIC para K in [2, 5].
    3. Entrenamiento del GMM de Regímenes Macroeconómicos (K=3) y probabilidades a posteriori.
    4. Entrenamiento del GMM de Arquetipos Corporativos (K=4) y probabilidades a posteriori.
    5. Validación de consistencia estadística y métricas de insolvencia.
    6. Persistencia de modelos, reporte JSON y actualización del dataset Parquet.
    """
    print("=" * 95)
    print("SUBAGENTE 1A: REGÍMENES MACROECONÓMICOS Y ARQUETIPOS CORPORATIVOS (GMM CLUSTERING)")
    print("=" * 95)

    if not input_parquet_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset Parquet en: {input_parquet_path}")

    # 1. Cargar Dataset
    logger.info("Cargando dataset desde: %s", input_parquet_path)
    df = pd.read_parquet(input_parquet_path)
    total_obs = len(df)
    logger.info("Observaciones cargadas: %s filas, %s columnas.", f"{total_obs:,}", len(df.columns))

    # 2. Macro GMM: Evaluación BIC y Entrenamiento
    print("\n" + "-" * 80)
    print("FASE 1: MODELO DE MEZCLA GAUSSIANA MACROECONÓMICA (REGÍMENES DE MERCADO)")
    print("-" * 80)
    
    macro_detector = MacroGMMRegimeDetector(n_components=3, covariance_type="full", random_state=42, n_init=10)
    macro_bic_df = macro_detector.evaluate_bic(df, k_range=range(2, 6), cov_types=["full", "diag"])
    
    print("\n--- Resultados Criterio de Información Bayesiano (BIC) Macro ---")
    print(macro_bic_df.to_string(index=False))

    macro_detector.fit(df)
    df_transformed = macro_detector.transform(df)

    print("\n--- Centros de los Regímenes Macroeconómicos (Escala Original) ---")
    print(macro_detector.regime_means_.to_string())

    print("\n--- Pesos de Mezcla por Régimen ---")
    for r_idx, w in enumerate(macro_detector.regime_weights_):
        print(f"  * Régimen {r_idx} ({MacroGMMRegimeDetector.REGIME_NAMES[r_idx]}): {w * 100:.2f}%")

    # 3. Corporate Archetypes GMM: Evaluación BIC y Entrenamiento
    print("\n" + "-" * 80)
    print("FASE 2: MODELO DE MEZCLA GAUSSIANA CORPORATIVA (ARQUETIPOS DE SOLVENCIA Y LIQUIDEZ)")
    print("-" * 80)

    corp_detector = CorporateArchetypeGMMDetector(n_components=4, covariance_type="full", random_state=42, n_init=5)
    corp_bic_df = corp_detector.evaluate_bic(df, k_range=range(2, 6), cov_types=["full", "diag"], sample_size=50000)

    print("\n--- Resultados Criterio de Información Bayesiano (BIC) Corporativo ---")
    print(corp_bic_df.to_string(index=False))

    corp_detector.fit(df, sample_size=100000)
    df_final = corp_detector.transform(df_transformed)

    print("\n--- Centros de los Arquetipos Corporativos (Escala Original en USD) ---")
    print(corp_detector.archetype_means_.to_string())

    print("\n--- Pesos de Mezcla por Arquetipo Corporativo ---")
    for a_idx, w in enumerate(corp_detector.archetype_weights_):
        print(f"  * Arquetipo {a_idx} ({CorporateArchetypeGMMDetector.ARCHETYPE_NAMES[a_idx]}): {w * 100:.2f}%")

    # 4. Validación de Consistencia y Métricas Cruzadas
    print("\n" + "-" * 80)
    print("FASE 3: VALIDACIÓN ESTADÍSTICA Y TASAS DE INSOLVENCIA HISTÓRICA")
    print("-" * 80)

    # Validar sumas de probabilidad
    macro_prob_cols = ["macro_regime_expansion_prob", "macro_regime_neutral_prob", "macro_regime_crisis_prob"]
    corp_prob_cols = [f"corp_archetype_prob_{i}" for i in range(4)]

    macro_prob_sum = df_final[macro_prob_cols].sum(axis=1)
    corp_prob_sum = df_final[corp_prob_cols].sum(axis=1)

    macro_sum_valid = np.allclose(macro_prob_sum, 1.0, atol=1e-4)
    corp_sum_valid = np.allclose(corp_prob_sum, 1.0, atol=1e-4)

    logger.info("Validación Probabilidades Macro (Suma == 1.0): %s (Min=%.5f, Max=%.5f)",
                "CORRECTO" if macro_sum_valid else "ERROR", macro_prob_sum.min(), macro_prob_sum.max())
    logger.info("Validación Probabilidades Corp (Suma == 1.0): %s (Min=%.5f, Max=%.5f)",
                "CORRECTO" if corp_sum_valid else "ERROR", corp_prob_sum.min(), corp_prob_sum.max())

    # Tasa de bancarrota por régimen y arquetipo si la columna de target existe
    target_col = "target_bankrupt_12m" if "target_bankrupt_12m" in df_final.columns else None
    
    macro_target_stats = {}
    if target_col:
        print("\n--- Tasa de Insolvencia a 12 meses por Régimen Macroeconómico ---")
        regime_stat = df_final.groupby("macro_regime_label")[target_col].agg(["count", "sum", "mean"]).rename(
            columns={"count": "N_Observaciones", "sum": "Total_Quiebras", "mean": "Tasa_Insolvencia_12M"}
        )
        regime_stat["Regime_Name"] = [MacroGMMRegimeDetector.REGIME_NAMES[r] for r in regime_stat.index]
        print(regime_stat.to_string())
        macro_target_stats = regime_stat.to_dict(orient="index")

    corp_target_stats = {}
    if target_col:
        print("\n--- Tasa de Insolvencia a 12 meses por Arquetipo Corporativo ---")
        archetype_stat = df_final.groupby("corp_archetype_label")[target_col].agg(["count", "sum", "mean"]).rename(
            columns={"count": "N_Observaciones", "sum": "Total_Quiebras", "mean": "Tasa_Insolvencia_12M"}
        )
        archetype_stat["Archetype_Name"] = [CorporateArchetypeGMMDetector.ARCHETYPE_NAMES[a] for a in archetype_stat.index]
        print(archetype_stat.to_string())
        corp_target_stats = archetype_stat.to_dict(orient="index")

    # 5. Persistencia de Modelos
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if save_models:
        macro_model_path = MODELS_DIR / "macro_gmm_regimes_model.pkl"
        corp_model_path = MODELS_DIR / "corp_archetypes_gmm_model.pkl"

        joblib.dump(macro_detector, macro_model_path)
        joblib.dump(corp_detector, corp_model_path)
        logger.info("Modelos GMM guardados en:\n  - %s\n  - %s", macro_model_path, corp_model_path)

    # 6. Guardar Reporte de Auditoría JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_dict = {
        "dataset_path": str(input_parquet_path),
        "total_observations": int(total_obs),
        "macro_gmm": {
            "n_components": macro_detector.n_components,
            "covariance_type": macro_detector.covariance_type,
            "bic_table": macro_bic_df.to_dict(orient="records"),
            "regime_weights": {f"regime_{r}": float(w) for r, w in enumerate(macro_detector.regime_weights_)},
            "regime_means": macro_detector.regime_means_.to_dict(orient="index"),
            "target_distribution_12m": {str(k): {m: float(v) if isinstance(v, (int, float, np.number)) else v for m, v in vals.items()} for k, vals in macro_target_stats.items()}
        },
        "corporate_gmm": {
            "n_components": corp_detector.n_components,
            "covariance_type": corp_detector.covariance_type,
            "bic_table": corp_bic_df.to_dict(orient="records"),
            "archetype_weights": {f"archetype_{a}": float(w) for a, w in enumerate(corp_detector.archetype_weights_)},
            "archetype_means": corp_detector.archetype_means_.to_dict(orient="index"),
            "target_distribution_12m": {str(k): {m: float(v) if isinstance(v, (int, float, np.number)) else v for m, v in vals.items()} for k, vals in corp_target_stats.items()}
        },
        "validation": {
            "macro_probabilities_valid_sum": bool(macro_sum_valid),
            "corp_probabilities_valid_sum": bool(corp_sum_valid),
            "new_features_generated": macro_prob_cols + ["macro_regime_label", "macro_regime_name"] + corp_prob_cols + ["corp_archetype_label", "corp_archetype_name"]
        }
    }

    if save_report:
        report_path = REPORTS_DIR / "macro_gmm_regimes_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        logger.info("Informe JSON de auditoría guardado en: %s", report_path)

    # 7. Actualización del Parquet Máster
    target_out_path = output_parquet_path or input_parquet_path
    logger.info("Guardando dataset enriquecido con características GMM en: %s", target_out_path)
    df_final.to_parquet(target_out_path, index=False, compression="snappy")
    
    # También actualizar v2_sec_financials_pivoted_clean.parquet si existe
    if DEFAULT_CLEAN_PARQUET_PATH.exists() and target_out_path != DEFAULT_CLEAN_PARQUET_PATH:
        try:
            clean_df = pd.read_parquet(DEFAULT_CLEAN_PARQUET_PATH)
            # Agregar las nuevas columnas si no existen
            cols_to_add = [c for c in df_final.columns if c in (macro_prob_cols + ["macro_regime_label", "macro_regime_name"] + corp_prob_cols + ["corp_archetype_label", "corp_archetype_name"])]
            for col in cols_to_add:
                clean_df[col] = df_final[col]
            clean_df.to_parquet(DEFAULT_CLEAN_PARQUET_PATH, index=False, compression="snappy")
            logger.info("Dataset limpio actualizado exitosamente: %s", DEFAULT_CLEAN_PARQUET_PATH)
        except Exception as ex:
            logger.warning("No se pudo sincronizar clean parquet: %s", ex)

    print("\n" + "=" * 95)
    print("PIPELINE DE REGÍMENES MACRO Y ARQUETIPOS GMM COMPLETADO SATISFACTORIAMENTE")
    print("=" * 95)

    return df_final, report_dict


if __name__ == "__main__":
    run_macro_corporate_gmm_pipeline()
