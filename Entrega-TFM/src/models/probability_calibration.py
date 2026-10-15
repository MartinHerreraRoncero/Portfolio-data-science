"""
Módulo de Calibración de Probabilidades de Incumplimiento (PD) bajo Estándares Regulatorios (IFRS 9 / Basel III).
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def calibrate_credit_model(model, X_calib, y_calib, X_test, y_test, method='sigmoid', cv_mode=5, figure_dir="reports/figures", show=False):
    """
    Calibra las probabilidades de salida de un modelo crediticio utilizando Platt Scaling (Sigmoid)
    o Regresión Isotónica, generando curva de calibración para auditoría regulatoria IFRS 9.
    Si el estimador ya está entrenado y se pasa un conjunto de validación independiente (X_calib),
    se utiliza cv='prefit' sobre dicho conjunto independiente. De lo contrario, se usa validación cruzada interna cv=5.
    """
    if cv_mode == 'prefit':
        calibrated_model = CalibratedClassifierCV(estimator=model, method=method, cv='prefit')
        calibrated_model.fit(X_calib, y_calib)
    else:
        calibrated_model = CalibratedClassifierCV(estimator=model, method=method, cv=cv_mode)
        calibrated_model.fit(X_calib, y_calib)
    
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(X_test)[:, 1]
    else:
        raw_probs = model(X_test)
        
    cal_probs = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Gráficos de calibración de probabilidades
    fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(y_test, raw_probs, n_bins=10)
    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_test, cal_probs, n_bins=10)
    
    os.makedirs(figure_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectamente Calibrado")
    plt.plot(mean_predicted_value_raw, fraction_of_positives_raw, "s-", label="Sin Calibrar")
    plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, "o-", label=f"Calibrado ({method.capitalize()})")
    plt.ylabel("Fracción Real de Quiebras ($y=1$)")
    plt.xlabel("Probabilidad Predicha ($\hat{P}(Y=1)$)")
    plt.title("Curva de Calibración de Probabilidades de Insolvencia (IFRS 9 / Basel III)")
    plt.legend(loc="lower right")
    plt.grid(True)
    out_path = os.path.join(figure_dir, "fig5_probability_calibration.png")
    plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"[INFO] Modelo crediticio calibrado exitosamente mediante Platt Scaling ({method}). Curva guardada en {out_path}.")
    return calibrated_model, cal_probs
