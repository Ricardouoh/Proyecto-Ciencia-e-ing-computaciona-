from __future__ import annotations
"""
Entrenamiento supervisado de modelos clínicos (MLP / Regresión Logística).

FUNCIONALIDAD:
- Carga los datasets preprocesados desde:
      data/processed/train.csv
      data/processed/val.csv
- Construye el modelo desde:
      src.model.make_model(cfg)
- Soporta:
      • Regresión Logística (fit directo)
      • MLPClassifier con:
            - entrenamiento por épocas (warm_start)
            - early stopping manual
            - balanceo automático por clase
- Calcula métricas en validación:
      • AUROC
      • AUPRC
- Guarda automáticamente:
      • results/model.joblib        -> mejor modelo
      • results/train_log.csv      -> métricas por época

COMANDOS PARA EJECUCIÓN:

>>> Entrenar Regresión Logística (baseline, se recomienda primero):
python -m src.train_loop --data-dir data/processed --model logreg

>>> Entrenar MLP:
python -m src.train_loop --data-dir data/processed --model mlp

SALIDA ESPERADA:
results/
 ├── model.joblib       (modelo entrenado)
 └── train_log.csv      (log de entrenamiento)

REQUISITOS:
- Haber ejecutado previamente:
      python -m src.preprocess --csv data/raw.csv --target label --outdir data/processed
- Deben existir:
      data/processed/train.csv
      data/processed/val.csv
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.model import make_model, default_config
from src import metrics as mt


# =========================
# CONFIGURACIÓN
# =========================

@dataclass
class TrainConfig:
    model_cfg: Dict[str, Any]
    max_epochs: int = 100
    patience: int = 10
    outdir: Path = Path("results")


# =========================
# CARGA DE DATOS
# =========================

def _load_xy(csv_path: Path, target: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"No se encuentra la columna {target} en {csv_path}")
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y


# =========================
# MÉTRICAS
# =========================

def _eval_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        out["auroc"] = roc_auc_score(y_true, proba)
    except Exception:
        out["auroc"] = float("nan")

    try:
        out["auprc"] = average_precision_score(y_true, proba)
    except Exception:
        out["auprc"] = float("nan")

    return out


def _save_log(log_rows, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(out_csv, index=False)


def _save_model(model, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def _class_weight_sample_weights(y: pd.Series):
    """
    Devuelve sample_weight balanceado por clase para modelos que lo soporten.
    Si solo hay una clase, devuelve None.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    total = len(y)
    weights = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weights[int(t)] for t in y.tolist()])


# =========================
# ENTRENAMIENTO PRINCIPAL
# =========================

def train_and_validate(cfg: TrainConfig, data_dir: Path, target: str = "label") -> None:
    # Cargar datos
    Xtr, ytr = _load_xy(data_dir / "train.csv", target=target)
    Xva, yva = _load_xy(data_dir / "val.csv", target=target)

    # Construir modelo
    model = make_model(cfg.model_cfg)

    # Paths de salida
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.outdir / "model.joblib"
    log_path = cfg.outdir / "train_log.csv"

    log_rows = []

    # ===== Regresión Logística (u otro modelo no incremental) =====
    if model.__class__.__name__.lower().startswith("logistic"):
        model.fit(Xtr, ytr)
        proba_va = model.predict_proba(Xva)[:, 1]

        metrics = _eval_metrics(yva.values, proba_va)
        log_rows.append({"epoch": 1, **metrics})

        _save_log(log_rows, log_path)
        _save_model(model, model_path)

        full_metrics = mt.compute_classification_metrics(yva.values, proba_va)
        mt.save_json(full_metrics, cfg.outdir / "metrics_val.json")
        mt.plot_roc(yva.values, proba_va, cfg.outdir / "roc_val.png")
        mt.plot_pr(yva.values, proba_va, cfg.outdir / "pr_val.png")
        preds_va = (proba_va >= 0.5).astype(int)
        mt.plot_confusion_matrix(yva.values, preds_va, cfg.outdir / "confusion_matrix_val.png")

        print("OK Modelo guardado:", model_path)
        print(f"OK Metricas validacion: AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f}")
        return

    # ===== MLP con entrenamiento por épocas + early stopping =====
    # Configuración incremental
    params = model.get_params()
    if params.get("max_iter", 1) != 1:
        model.set_params(max_iter=1)
    if not params.get("warm_start", False):
        try:
            model.set_params(warm_start=True)
        except Exception:
            pass

    sw = _class_weight_sample_weights(ytr)

    best_auroc = -np.inf
    best_model = None
    no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        # Entrenar una "época"
        try:
            if sw is not None:
                model.fit(Xtr, ytr, sample_weight=sw)
            else:
                model.fit(Xtr, ytr)
        except TypeError:
            # Si no acepta sample_weight
            model.fit(Xtr, ytr)

        # Evaluar en validación
        proba_va = model.predict_proba(Xva)[:, 1]
        metrics = _eval_metrics(yva.values, proba_va)
        metrics["epoch"] = epoch
        log_rows.append(metrics)

        auroc = metrics["auroc"]
        auprc = metrics["auprc"]
        print(f"Epoch {epoch:03d} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

        # Early stopping (mejor AUROC)
        if np.isfinite(auroc) and auroc > best_auroc:
            best_auroc = auroc
            best_model = copy.deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            print("STOP Early stopping activado")
            break

    # Guardar log
    _save_log(log_rows, log_path)

    # Guardar mejor modelo
    final_model = best_model if best_model is not None else model
    _save_model(final_model, model_path)

    proba_va = final_model.predict_proba(Xva)[:, 1]
    final_metrics = mt.compute_classification_metrics(yva.values, proba_va)
    mt.save_json(final_metrics, cfg.outdir / "metrics_val.json")
    mt.plot_roc(yva.values, proba_va, cfg.outdir / "roc_val.png")
    mt.plot_pr(yva.values, proba_va, cfg.outdir / "pr_val.png")
    preds_va = (proba_va >= 0.5).astype(int)
    mt.plot_confusion_matrix(yva.values, preds_va, cfg.outdir / "confusion_matrix_val.png")

    print("OK Mejor modelo guardado en:", model_path)
    print("LOG Log de entrenamiento:", log_path)
    print(f"OK Metricas validacion (mejor modelo): AUROC={final_metrics['auroc']:.4f} | AUPRC={final_metrics['auprc']:.4f}")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser(description="Entrenamiento de modelos clínicos")
    ap.add_argument("--data-dir", default="data/processed", help="Carpeta con train.csv y val.csv")
    ap.add_argument("--model", default="mlp", choices=["mlp", "logreg"], help="Tipo de modelo a entrenar")
    args = ap.parse_args()

    # Soporta default_config como función o como diccionario
    if callable(default_config):
        model_cfg = default_config(args.model)
    else:
        model_cfg = default_config[args.model]

    cfg = TrainConfig(model_cfg=model_cfg)
    train_and_validate(cfg, Path(args.data_dir))
