from __future__ import annotations
"""
Métricas y gráficos para clasificación binaria.

Funciones:
- compute_classification_metrics: AUROC, AUPRC y métricas a umbral (acc/prec/recall/F1)
- confusion_counts: TP, FP, TN, FN
- plot_roc: guarda curva ROC
- plot_pr: guarda curva Precision-Recall
- save_json: escribe un dict en JSON
"""

from pathlib import Path
from typing import Dict, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt


def compute_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calcula AUROC, AUPRC y métricas a un umbral dado."""
    out: Dict[str, float] = {}

    # Probabilidades bien formadas
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    # Métricas umbraladas
    y_pred = (y_proba >= float(threshold)).astype(int)

    # Métricas robustas (maneja excepciones si solo hay una clase)
    try:
        out["auroc"] = roc_auc_score(y_true, y_proba)
    except Exception:
        out["auroc"] = float("nan")

    try:
        out["auprc"] = average_precision_score(y_true, y_proba)
    except Exception:
        out["auprc"] = float("nan")

    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)

    return out


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """Retorna TP, FP, TN, FN en un dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def plot_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
) -> None:
    """Guarda curva ROC en out_path (PNG)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float("nan")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
) -> None:
    """Guarda curva Precision-Recall en out_path (PNG)."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    try:
        auprc = average_precision_score(y_true, y_proba)
    except Exception:
        auprc = float("nan")

    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def save_json(data: Dict, out_path: Path) -> None:
    """Guarda un diccionario en JSON (con indentación)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def table_from_metrics(d: Dict[str, float]) -> pd.DataFrame:
    """Convierte el dict de métricas a DataFrame de una fila (útil para logs)."""
    return pd.D
