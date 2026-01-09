from __future__ import annotations
"""
Evaluación en TEST:
- Carga results/model.joblib
- Lee data/processed/test.csv
- Calcula AUROC/AUPRC y métricas a umbral
- Guarda ROC.png, PR.png y test_metrics.json
"""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.metrics import (
    compute_classification_metrics,
    confusion_counts,
    plot_confusion_matrix,
    plot_pr,
    plot_roc,
    save_json,
)


def evaluate_model(
    model_path: Path,
    test_csv: Path,
    outdir: Path,
    target: str = "label",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Carga modelo + test, calcula métricas y guarda gráficos."""
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    if target not in df.columns:
        raise ValueError(f"No se encontró la columna target '{target}' en {test_csv}")

    y_true = df[target].astype(int).values
    X = df.drop(columns=[target])

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = compute_classification_metrics(y_true, proba, threshold=threshold)
    metrics.update(confusion_counts(y_true, preds))
    metrics["threshold"] = threshold
    metrics["n_rows"] = int(len(df))

    outdir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, outdir / "test_metrics.json")
    plot_roc(y_true, proba, outdir / "roc_test.png")
    plot_pr(y_true, proba, outdir / "pr_test.png")
    plot_confusion_matrix(y_true, preds, outdir / "confusion_matrix_test.png")

    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evalúa el modelo guardado en el split de test.")
    ap.add_argument("--model", default="results/model.joblib", help="Ruta al modelo entrenado.")
    ap.add_argument("--test-csv", default="data/processed/test.csv", help="CSV de test (features + label).")
    ap.add_argument("--outdir", default="results", help="Carpeta de salida para métricas y gráficos.")
    ap.add_argument("--target", default="label", help="Nombre de la columna target.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Umbral para métricas umbraladas.")
    args = ap.parse_args()

    metrics = evaluate_model(
        model_path=Path(args.model),
        test_csv=Path(args.test_csv),
        outdir=Path(args.outdir),
        target=args.target,
        threshold=float(args.threshold),
    )

    print("OK Metricas de test guardadas en:", Path(args.outdir) / "test_metrics.json")
    print(f"   AUROC={metrics.get('auroc'):.4f} | AUPRC={metrics.get('auprc'):.4f} | Acc={metrics.get('accuracy'):.4f}")
