from __future__ import annotations
"""
Evaluación del mejor modelo en el conjunto de test.

- Carga:
    * results/model.joblib
    * data/processed/test.csv
- Calcula:
    * AUROC
    * AUPRC
    * accuracy
- Guarda:
    * results/test_metrics.json
    * results/test_roc_<target>.png
    * results/test_pr_<target>.png
    * results/test_confusion_<target>.png
    * results/train_log_curves_<target>.png (si existe results/train_log.csv)
    * results/reg_pred_vs_actual_<target>.png (regresion)
    * results/reg_residuals_<target>.png (regresion)
    
    
    
    
SALIDA DEL MÓDULO DE EVALUACIÓN:

1) Métricas impresas por consola:

   - AUROC (Area Under the ROC Curve):
     Mide la capacidad del modelo para distinguir entre pacientes de la clase
     positiva y la clase negativa en todos los posibles umbrales de decisión.
     Valores cercanos a 1 indican excelente capacidad de discriminación.
     Un valor cercano a 0.5 indica un modelo sin poder predictivo.

   - AUPRC (Area Under the Precision-Recall Curve):
     Evalúa el compromiso entre precisión y exhaustividad (recall),
     especialmente útil cuando las clases están desbalanceadas.
     Valores altos indican que el modelo identifica correctamente los
     casos positivos con pocos falsos positivos.

   - Accuracy (Exactitud):
     Porcentaje total de predicciones correctas realizadas por el modelo
     sobre el conjunto de prueba.
     Puede ser engañosa en conjuntos de datos desbalanceados, por lo que
     debe interpretarse junto con AUROC y AUPRC.

   - n_test:
     Cantidad de muestras utilizadas en la evaluación final del modelo.

2) Archivo opcional: results/test_metrics.json

   Contiene:
   - El valor del umbral utilizado para clasificar (threshold)
   - Las métricas AUROC, AUPRC y Accuracy
   - El número total de muestras evaluadas (n_test)
   Este archivo permite conservar resultados para reportes, gráficos y
   comparación entre distintos modelos entrenados.

------------------------------------------------------------

INTERPRETACIÓN DE LOS RESULTADOS:

Las métricas obtenidas en esta etapa reflejan el DESEMPEÑO REAL del modelo
en un escenario clínico simulado, ya que:

- El conjunto de prueba (test.csv) nunca fue utilizado en el proceso de
  entrenamiento ni durante la etapa de validación.
- Esto evita el sobreajuste artificial y permite estimar la capacidad real
  de generalización del modelo sobre pacientes no vistos.
- Valores altos de AUROC y AUPRC indican que el modelo presenta un alto poder
  predictivo y una excelente capacidad de discriminación clínica.
- Si las métricas fueran bajas, significaría que el modelo es inestable,
  poco generalizable o que requiere mejorar el preprocesamiento,
  la arquitectura o el balance de clases.

En resumen, estas métricas representan la calidad final del sistema de
predicción clínica construido en el proyecto.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)


def _load_xy(
    csv_path: Path,
    target: str = "label",
    task: str = "classification",
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"No se encuentra la columna {target} en {csv_path}")
    if task == "classification":
        y = df[target].astype(int)
        X = df.drop(columns=[target])
        return X, y
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    y = df[target].astype(float)
    X = df.drop(columns=[target])
    return X, y


def _safe_target_name(target: str) -> str:
    if not target:
        return "target"
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in target)


def _name_with_target(base_name: str, target: str) -> str:
    p = Path(base_name)
    safe = _safe_target_name(target)
    return f"{p.stem}_{safe}{p.suffix}"


def _plot_train_curves(log_csv: Path, out_dir: Path, target: str) -> None:
    if not log_csv.exists():
        return
    df = pd.read_csv(log_csv)
    if "epoch" not in df.columns:
        return
    if "auroc" not in df.columns and "auprc" not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    if "auroc" in df.columns:
        plt.plot(df["epoch"], df["auroc"], label="AUROC")
    if "auprc" in df.columns:
        plt.plot(df["epoch"], df["auprc"], label="AUPRC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Training metrics")
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _name_with_target("train_log_curves.png", target)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_roc_pr_confusion(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    target: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC (test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / _name_with_target("test_roc.png", target), dpi=150)
        plt.close()
    except Exception:
        pass

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auprc = average_precision_score(y_true, y_proba)
        plt.figure(figsize=(5, 4))
        plt.plot(recall, precision, label=f"AUPRC={auprc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR (test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / _name_with_target("test_pr.png", target), dpi=150)
        plt.close()
    except Exception:
        pass

    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4.5, 4))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion matrix (test)")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0", "1"])
        plt.yticks(tick_marks, ["0", "1"])
        thresh = cm.max() / 2.0 if cm.max() else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(out_dir / _name_with_target("test_confusion.png", target), dpi=150)
        plt.close()
    except Exception:
        pass


def _plot_regression_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    target: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predicho vs real (colores por signo del residuo)
    residuals = y_true - y_pred
    pos_mask = residuals >= 0
    neg_mask = ~pos_mask
    plt.figure(figsize=(5.5, 4))
    if neg_mask.any():
        plt.scatter(y_true[neg_mask], y_pred[neg_mask], s=18, alpha=0.7, color="#c44e52", label="Sobreestima (residuo < 0)")
    if pos_mask.any():
        plt.scatter(y_true[pos_mask], y_pred[pos_mask], s=18, alpha=0.7, color="#4c72b0", label="Subestima (residuo >= 0)")
    min_v = float(np.nanmin([y_true.min(), y_pred.min()]))
    max_v = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs actual (test)")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / _name_with_target("reg_pred_vs_actual.png", target), dpi=150)
    plt.close()

    # Residuales
    plt.figure(figsize=(5.5, 4))
    if neg_mask.any():
        plt.hist(residuals[neg_mask], bins=15, color="#c44e52", alpha=0.75, label="Sobreestima (residuo < 0)")
    if pos_mask.any():
        plt.hist(residuals[pos_mask], bins=15, color="#4c72b0", alpha=0.75, label="Subestima (residuo >= 0)")
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Count")
    plt.title("Residuals (test)")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / _name_with_target("reg_residuals.png", target), dpi=150)
    plt.close()


def evaluate_test(
    model_path: str | Path = "results/model.joblib",
    test_csv: str | Path = "data/processed/test.csv",
    out_json: str | Path = "results/test_metrics.json",
    target: str = "label",
    task: str = "classification",
) -> Dict:
    model_path = Path(model_path)
    test_csv = Path(test_csv)
    out_json = Path(out_json)

    if not model_path.exists():
        raise FileNotFoundError(f"No encontré el modelo en {model_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"No encontré el test CSV en {test_csv}")

    model = joblib.load(model_path)
    Xte, yte = _load_xy(test_csv, target=target, task=task)

    metrics = {}
    if task == "classification":
        proba = model.predict_proba(Xte)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        try:
            metrics["auroc"] = roc_auc_score(yte, proba)
        except Exception:
            metrics["auroc"] = float("nan")
        try:
            metrics["auprc"] = average_precision_score(yte, proba)
        except Exception:
            metrics["auprc"] = float("nan")

        metrics["accuracy"] = accuracy_score(yte, y_pred)
        metrics["n_test"] = int(len(yte))
    else:
        preds = model.predict(Xte)
        try:
            metrics["r2"] = r2_score(yte, preds)
        except Exception:
            metrics["r2"] = float("nan")
        metrics["mae"] = mean_absolute_error(yte, preds)
        metrics["rmse"] = float(np.sqrt(mean_squared_error(yte, preds)))
        metrics["n_test"] = int(len(yte))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if task == "classification":
        _plot_roc_pr_confusion(yte.values, proba, y_pred, out_json.parent, target)
        _plot_train_curves(Path("results/train_log.csv"), out_json.parent, target)
    else:
        _plot_regression_outputs(yte.values, preds, out_json.parent, target)

    print("===== MÉTRICAS EN TEST =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("============================")

    return metrics


def print_summary(results_dir: Path) -> None:
    metrics_path = results_dir / "test_metrics.json"
    threshold_path = results_dir / "threshold.json"

    if not metrics_path.exists():
        print("No se encontro test_metrics.json en:", metrics_path)
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("===== RESUMEN DEL MODELO =====")
    if "global" in data:
        g = data["global"]
        print("Global:")
        for k in ("auroc", "auprc", "accuracy", "precision", "recall", "f1"):
            if k in g:
                print(f"  {k}: {g[k]}")
        if "n_rows" in g:
            print(f"  n_rows: {g['n_rows']}")
    else:
        for k, v in data.items():
            if isinstance(v, (int, float)):
                print(f"{k}: {v}")

    if "by_domain" in data and data["by_domain"]:
        print("Por dominio:")
        for dom, m in data["by_domain"].items():
            line = f"  {dom}: recall={m.get('recall')} precision={m.get('precision')} n={m.get('n_rows')}"
            print(line)

    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            th = json.load(f)
        print("Threshold:")
        print(f"  threshold: {th.get('threshold')}")
        print(f"  metric: {th.get('metric')}")
        if th.get("min_precision") is not None:
            print(f"  min_precision: {th.get('min_precision')}")
    print("==============================")


if __name__ == "__main__":
    import argparse

<<<<<<< HEAD
    ap = argparse.ArgumentParser(description="Resumen de entrenamiento y evaluacion")
    ap.add_argument("--model-path", default="results/model.joblib")
    ap.add_argument("--test-csv", default=None)
    ap.add_argument("--out-json", default="results/test_metrics.json")
    ap.add_argument("--target", default="label")
    ap.add_argument("--task", choices=["classification", "regression"], default="classification")
    ap.add_argument("--data-dir", default=None, help="Carpeta con aligned_test.csv o test.csv.")
    ap.add_argument("--results-dir", default="results", help="Carpeta con resultados.")
    ap.add_argument("--summary-only", action="store_true", help="Solo imprime resumen y sale.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if args.data_dir:
        print_summary(results_dir)
        if args.summary_only:
            raise SystemExit(0)

        data_dir = Path(args.data_dir)
        aligned_test = data_dir / "aligned_test.csv"
        test_csv = aligned_test if aligned_test.exists() else (data_dir / "test.csv")
    else:
        test_csv = Path(args.test_csv) if args.test_csv else Path("data/processed/test.csv")

    evaluate_test(
        model_path=Path(args.model_path),
        test_csv=Path(test_csv),
        out_json=Path(args.out_json),
=======
    ap = argparse.ArgumentParser(description="Evaluacion en test")
    ap.add_argument("--model-path", default="results/model.joblib")
    ap.add_argument("--test-csv", default="data/processed/test.csv")
    ap.add_argument("--out-json", default="results/test_metrics.json")
    ap.add_argument("--target", default="label")
    ap.add_argument("--task", choices=["classification", "regression"], default="classification")
    args = ap.parse_args()

    evaluate_test(
        model_path=args.model_path,
        test_csv=args.test_csv,
        out_json=args.out_json,
>>>>>>> 56aae0ceea2bf5a2765e2543d64f0e22b032b50e
        target=args.target,
        task=args.task,
    )
