"""
Módulo de Función de Pérdida Asimétrica de Negocio Financiero.
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.

Matriz de Costes de Negocio Financiero (EAD = $10,000,000 USD):
- Net Interest Margin (NIM) = 2.5% -> Margen Comercial = $250,000 USD
- Loss Given Default (LGD) = 60.0% -> Pérdida por Insolvencia = $6,000,000 USD
- Coste de Falso Positivo (CFP) = $250,000 USD (Margen perdido por denegación injustificada)
- Coste de Falso Negativo (CFN) = $6,000,000 USD (Pérdida neta de capital por insolvencia)
- Ratio de Asimetría: CFN / CFP = 6,000,000 / 250,000 = 24.0x
"""

import numpy as np
import torch
import torch.nn as nn

# Parámetros Globales de Exposición Financiera
EAD_USD = 10000000.0  # Exposición $10M USD por préstamo
NIM_PCT = 0.025       # Margen de interés neto 2.5%
LGD_PCT = 0.600       # Pérdida en caso de incumplimiento 60.0%

COST_FP_USD = EAD_USD * NIM_PCT   # $250,000 USD
COST_FN_USD = EAD_USD * LGD_PCT   # $6,000,000 USD
PROFIT_TN_USD = EAD_USD * NIM_PCT # +$250,000 USD
AVOIDED_TP_USD = EAD_USD * LGD_PCT# +$6,000,000 USD

ASYMMETRY_RATIO = COST_FN_USD / COST_FP_USD  # 24.0x


class AsymmetricLossPyTorch(nn.Module):
    """
    Función de pérdida asimétrica de negocio para PyTorch basada en logits continuos.
    Formulación analítica exacta y numéricamente estable:
    L(z, y) = y * C_FN * softplus(-z) + (1 - y) * C_FP * softplus(z)
    donde z son los logits directos (sin capa Sigmoid previa) y y in {0, 1}.
    Penaliza los Falsos Negativos 24x más que los Falsos Positivos sin riesgo de desbordamiento.
    """
    def __init__(self, cost_fn=24.0, cost_fp=1.0):
        super(AsymmetricLossPyTorch, self).__init__()
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def forward(self, logits, y_true):
        # logits: (N, 1) or (N,), y_true: (N, 1) or (N,)
        loss = y_true * self.cost_fn * nn.functional.softplus(-logits) + \
               (1.0 - y_true) * self.cost_fp * nn.functional.softplus(logits)
        return torch.mean(loss)



def compute_business_cost(y_true, y_pred_binary, cost_fn=COST_FN_USD, cost_fp=COST_FP_USD):
    """
    Calcula el coste total de negocio asociado a las clasificaciones erróneas.
    Coste Total = (FN * COST_FN_USD) + (FP * COST_FP_USD)
    """
    y_true = np.asarray(y_true)
    y_pred_binary = np.asarray(y_pred_binary)
    fn = int(np.sum((y_true == 1) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_binary == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred_binary == 1)))
    total_cost_usd = (fn * cost_fn) + (fp * cost_fp)
    return {
        'false_negatives': fn,
        'false_positives': fp,
        'true_negatives': tn,
        'true_positives': tp,
        'total_cost_usd': float(total_cost_usd),
        'fn': fn,
        'fp': fp,
        'tn': tn,
        'tp': tp
    }


def compute_detailed_financial_pnl(y_true, y_pred_binary):
    """
    Calcula el desglose completo del impacto financiero en USD (Coste Total y P&L Neto).
    """
    y_true = np.asarray(y_true)
    y_pred_binary = np.asarray(y_pred_binary)
    tn = int(np.sum((y_true == 0) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_binary == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred_binary == 1)))

    total_cost_usd = (fn * COST_FN_USD) + (fp * COST_FP_USD)
    net_pnl_usd = (tn * PROFIT_TN_USD) + (tp * AVOIDED_TP_USD) - (fp * COST_FP_USD) - (fn * COST_FN_USD)

    return {
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'total_cost_usd': float(total_cost_usd),
        'net_pnl_usd': float(net_pnl_usd)
    }

