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
      python -m src.preprocess --csv data/training/raw.csv --target label --outdir data/processed
- Deben existir:
      data/processed/train.csv
      data/processed/val.csv
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.model import make_model, default_config
from src import metrics as mt
from src.preprocess import infer_columns, make_preprocessor


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

def _load_xy(
    csv_path: Path,
    target: str = "label",
    task: str = "classification",
<<<<<<< HEAD
    domain_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"No se encuentra la columna {target} en {csv_path}")
    domain = None
    if domain_col and domain_col in df.columns:
        domain = df[domain_col].astype(str)
        df = df.drop(columns=[domain_col])
    if task == "classification":
        y = df[target].astype(int)
        X = df.drop(columns=[target])
        return X, y, domain
=======
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"No se encuentra la columna {target} en {csv_path}")
    if task == "classification":
        y = df[target].astype(int)
        X = df.drop(columns=[target])
        return X, y
>>>>>>> 56aae0ceea2bf5a2765e2543d64f0e22b032b50e
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    y = df[target].astype(float)
    X = df.drop(columns=[target])
    return X, y, domain


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


def _eval_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        out["r2"] = r2_score(y_true, y_pred)
    except Exception:
        out["r2"] = float("nan")
    out["mae"] = mean_absolute_error(y_true, y_pred)
    out["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
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


def _resolve_split_paths(data_dir: Path) -> Tuple[Path, Path, Path]:
    aligned_train = data_dir / "aligned_train.csv"
    aligned_val = data_dir / "aligned_val.csv"
    aligned_test = data_dir / "aligned_test.csv"
    if aligned_train.exists() and aligned_val.exists() and aligned_test.exists():
        return aligned_train, aligned_val, aligned_test
    return data_dir / "train.csv", data_dir / "val.csv", data_dir / "test.csv"


def _inject_missingness(
    X: pd.DataFrame,
    row_rate: float,
    col_rate: float,
    seed: int,
    hard_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    if row_rate <= 0 or col_rate <= 0:
        return X
    hard_cols = hard_cols or []
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    row_mask = rng.random(len(Xn)) < row_rate
    cols = list(Xn.columns)
    hard = set(hard_cols)
    for idx in Xn.index[row_mask]:
        for c in cols:
            rate = col_rate * (2.0 if c in hard else 1.0)
            rate = min(rate, 1.0)
            if rng.random() < rate:
                Xn.at[idx, c] = np.nan
    return Xn


def _fit_preprocessor(
    Xtr: pd.DataFrame,
    Xva: pd.DataFrame,
    Xte: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    out_path: Path,
    age_weight: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre = make_preprocessor(numeric_cols, categorical_cols, age_weight=age_weight)
    Xtr_t = pre.fit_transform(Xtr)
    Xva_t = pre.transform(Xva)
    Xte_t = pre.transform(Xte)

    if hasattr(pre, "get_feature_names_out"):
        feat_names = list(pre.get_feature_names_out())
    else:
        feat_names = [f"f{i}" for i in range(Xtr_t.shape[1])]

    Xtr_df = pd.DataFrame(Xtr_t, columns=feat_names, index=Xtr.index)
    Xva_df = pd.DataFrame(Xva_t, columns=feat_names, index=Xva.index)
    Xte_df = pd.DataFrame(Xte_t, columns=feat_names, index=Xte.index)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, out_path)
    return Xtr_df, Xva_df, Xte_df


def _scan_thresholds(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in np.linspace(0.0, 1.0, 101):
        metrics = mt.compute_classification_metrics(y_true, proba, threshold=float(t))
        metrics["threshold"] = float(t)
        metrics["pred_pos"] = int((proba >= t).sum())
        rows.append(metrics)
    return pd.DataFrame(rows)


def _choose_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric: str,
    min_precision: Optional[float],
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    scan = _scan_thresholds(y_true, proba)
    best = None
    for _, row in scan.iterrows():
        if metric == "recall" and min_precision is not None and row["precision"] < min_precision:
            continue
        if best is None or row[metric] > best[metric]:
            best = row
        elif best is not None and row[metric] == best[metric]:
            if metric == "recall" and row["precision"] > best["precision"]:
                best = row
    if best is None:
        best = scan.iloc[0]
    return float(best["threshold"]), best.to_dict(), scan

# =========================
# ENTRENAMIENTO PRINCIPAL
# =========================

def train_and_validate(
    cfg: TrainConfig,
    data_dir: Path,
    target: str = "label",
    task: str = "classification",
<<<<<<< HEAD
    optimize_threshold: bool = False,
    metric: str = "recall",
    min_precision: Optional[float] = None,
    domain_col: str = "domain",
    inject_missing: bool = False,
    missing_row_rate: float = 0.3,
    missing_col_rate: float = 0.2,
    missing_seed: int = 42,
    age_weight: float = 1.0,
) -> None:
    # Cargar datos (prioriza aligned_* si existen)
    train_path, val_path, test_path = _resolve_split_paths(data_dir)
    Xtr, ytr, _ = _load_xy(train_path, target=target, task=task, domain_col=domain_col)
    Xva, yva, _ = _load_xy(val_path, target=target, task=task, domain_col=domain_col)
    Xte, yte, domain_te = _load_xy(test_path, target=target, task=task, domain_col=domain_col)

    # Detectar columnas numericas/categoricas
    schema_path = data_dir / "schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        numeric_cols = [c for c in schema.get("numeric", []) if c in Xtr.columns]
        categorical_cols = [c for c in schema.get("categorical", []) if c in Xtr.columns]
    else:
        df_tmp = Xtr.copy()
        df_tmp[target] = ytr.values
        numeric_cols, categorical_cols = infer_columns(df_tmp, target=target)

    # Missingness injection (solo train)
    if inject_missing:
        before_missing = Xtr.isna().mean().mean()
        Xtr = _inject_missingness(
            Xtr,
            row_rate=missing_row_rate,
            col_rate=missing_col_rate,
            seed=missing_seed,
            hard_cols=["height_last", "weight_last", "bmi_last"],
        )
        after_missing = Xtr.isna().mean().mean()
        print(f"INFO Missingness train: {before_missing:.4f} -> {after_missing:.4f}")

    # Preprocesador (imputacion + escala + one-hot)
    pre_path = cfg.outdir / "preprocessor.joblib"
    Xtr, Xva, Xte = _fit_preprocessor(
        Xtr, Xva, Xte,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        out_path=pre_path,
        age_weight=age_weight,
    )
=======
) -> None:
    # Cargar datos
    Xtr, ytr = _load_xy(data_dir / "train.csv", target=target, task=task)
    Xva, yva = _load_xy(data_dir / "val.csv", target=target, task=task)
>>>>>>> 56aae0ceea2bf5a2765e2543d64f0e22b032b50e

    # Construir modelo
    model = make_model(cfg.model_cfg)

    # Paths de salida
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.outdir / "model.joblib"
    log_path = cfg.outdir / "train_log.csv"

    log_rows = []

    # ===== Regresion (no iterativa) =====
    if task == "regression":
        model.fit(Xtr, ytr)
        preds_va = model.predict(Xva)
        metrics = _eval_regression_metrics(yva.values, preds_va)
        log_rows.append({"epoch": 1, **metrics})

        _save_log(log_rows, log_path)
        _save_model(model, model_path)
        mt.save_json(metrics, cfg.outdir / "metrics_val.json")

        print("Modelo guardado:", model_path)
        print(f"Metricas validacion: R2={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}")
        return

<<<<<<< HEAD
    # ===== Regresion Logistica (no incremental) =====
=======
    # ===== Regresión Logística (u otro modelo no incremental) =====
>>>>>>> 56aae0ceea2bf5a2765e2543d64f0e22b032b50e
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

<<<<<<< HEAD
        print("OK Modelo guardado:", model_path)
        print(f"OK Metricas validacion: AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f}")
        final_model = model
=======
        print("Modelo guardado:", model_path)
        print(f"Metricas validacion: AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f}")
        return
>>>>>>> 56aae0ceea2bf5a2765e2543d64f0e22b032b50e

    else:
        # ===== MLP con entrenamiento por epocas + early stopping =====
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
            try:
                if sw is not None:
                    model.fit(Xtr, ytr, sample_weight=sw)
                else:
                    model.fit(Xtr, ytr)
            except TypeError:
                model.fit(Xtr, ytr)

            proba_va = model.predict_proba(Xva)[:, 1]
            metrics = _eval_metrics(yva.values, proba_va)
            metrics["epoch"] = epoch
            log_rows.append(metrics)

            auroc = metrics["auroc"]
            auprc = metrics["auprc"]
            print(f"Epoch {epoch:03d} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

            if np.isfinite(auroc) and auroc > best_auroc:
                best_auroc = auroc
                best_model = copy.deepcopy(model)
                no_improve = 0
            else:
                no_improve += 1

<<<<<<< HEAD
            if no_improve >= cfg.patience:
                print("STOP Early stopping activado")
                break
=======
        if no_improve >= cfg.patience:
            print("Early stopping activado")
            break
>>>>>>> 56aae0ceea2bf5a2765e2543d64f0e22b032b50e

        _save_log(log_rows, log_path)

        final_model = best_model if best_model is not None else model
        _save_model(final_model, model_path)

        proba_va = final_model.predict_proba(Xva)[:, 1]
        final_metrics = mt.compute_classification_metrics(yva.values, proba_va)
        mt.save_json(final_metrics, cfg.outdir / "metrics_val.json")
        mt.plot_roc(yva.values, proba_va, cfg.outdir / "roc_val.png")
        mt.plot_pr(yva.values, proba_va, cfg.outdir / "pr_val.png")
        preds_va = (proba_va >= 0.5).astype(int)
        mt.plot_confusion_matrix(yva.values, preds_va, cfg.outdir / "confusion_matrix_val.png")

<<<<<<< HEAD
        print("OK Mejor modelo guardado en:", model_path)
        print("LOG Log de entrenamiento:", log_path)
        print(f"OK Metricas validacion (mejor modelo): AUROC={final_metrics['auroc']:.4f} | AUPRC={final_metrics['auprc']:.4f}")
=======
    print("✔ Mejor modelo guardado en:", model_path)
    print("Log de entrenamiento:", log_path)
    print(f"Metricas validacion: AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f}")
>>>>>>> 56aae0ceea2bf5a2765e2543d64f0e22b032b50e

    # Seleccion de umbral
    threshold = 0.5
    best_val = None
    scan = _scan_thresholds(yva.values, proba_va)
    scan.to_csv(cfg.outdir / "threshold_scan.csv", index=False)
    if optimize_threshold:
        threshold, best_val, _ = _choose_threshold(
            yva.values, proba_va, metric=metric, min_precision=min_precision
        )
    threshold_info = {
        "threshold": threshold,
        "metric": metric,
        "min_precision": min_precision,
        "best_val": best_val,
    }
    mt.save_json(threshold_info, cfg.outdir / "threshold.json")

    # Evaluacion en test (global + por dominio)
    proba_te = final_model.predict_proba(Xte)[:, 1]
    preds_te = (proba_te >= threshold).astype(int)
    global_metrics = mt.compute_classification_metrics(yte.values, proba_te, threshold=threshold)
    global_metrics.update(mt.confusion_counts(yte.values, preds_te))
    global_metrics["threshold"] = threshold
    global_metrics["n_rows"] = int(len(yte))

    by_domain: Dict[str, Dict[str, float]] = {}
    if domain_te is not None:
        for dom in sorted(domain_te.unique()):
            mask = domain_te == dom
            yt = yte.values[mask]
            pt = proba_te[mask]
            pd = (pt >= threshold).astype(int)
            m = mt.compute_classification_metrics(yt, pt, threshold=threshold)
            m.update(mt.confusion_counts(yt, pd))
            m["threshold"] = threshold
            m["n_rows"] = int(mask.sum())
            by_domain[str(dom)] = m

    test_report = {
        "global": global_metrics,
        "by_domain": by_domain,
        "threshold": threshold,
    }
    mt.save_json(test_report, cfg.outdir / "test_metrics.json")

    mt.plot_roc(yte.values, proba_te, cfg.outdir / "roc_test.png")
    mt.plot_pr(yte.values, proba_te, cfg.outdir / "pr_test.png")
    mt.plot_confusion_matrix(yte.values, preds_te, cfg.outdir / "confusion_matrix_test.png")
# =========================
# CLI
# =========================

if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser(description="Entrenamiento de modelos clínicos")
    ap.add_argument("--data-dir", default="data/processed", help="Carpeta con train.csv y val.csv")
    ap.add_argument("--task", choices=["classification", "regression"], default="classification")
    ap.add_argument("--target", default="label", help="Columna objetivo.")
    ap.add_argument(
        "--model",
        default="mlp",
        choices=["mlp", "logreg", "gbr", "rf"],
        help="Tipo de modelo a entrenar",
    )
<<<<<<< HEAD
    ap.add_argument("--optimize-threshold", action="store_true", help="Optimiza umbral en validacion.")
    ap.add_argument(
        "--metric",
        default="recall",
        choices=["recall", "f1", "precision", "accuracy"],
        help="Metrica para seleccionar umbral (default=recall).",
    )
    ap.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Precision minima si metric=recall (opcional).",
    )
    ap.add_argument("--inject-missing", action="store_true", help="Inyecta missingness en train.")
    ap.add_argument("--missing-row-rate", type=float, default=0.3, help="Prob de filas con missing.")
    ap.add_argument("--missing-col-rate", type=float, default=0.2, help="Prob de NaN por columna.")
    ap.add_argument("--missing-seed", type=int, default=42, help="Seed para missingness.")
    ap.add_argument("--domain-col", default="domain", help="Columna de dominio para metricas.")
    ap.add_argument("--age-weight", type=float, default=1.0, help="Peso extra para age_years.")
=======
>>>>>>> 56aae0ceea2bf5a2765e2543d64f0e22b032b50e
    args = ap.parse_args()

    # Soporta default_config como función o como diccionario
    if callable(default_config):
        model_cfg = default_config(args.model)
    else:
        model_cfg = default_config[args.model]

    if args.task == "classification" and args.model in ("gbr", "rf"):
        raise ValueError("Modelo no valido para clasificacion. Usa 'logreg' o 'mlp'.")
    if args.task == "regression" and args.model in ("logreg", "mlp"):
        raise ValueError("Modelo no valido para regresion. Usa 'gbr' o 'rf'.")

    cfg = TrainConfig(model_cfg=model_cfg)
<<<<<<< HEAD
    train_and_validate(
        cfg,
        Path(args.data_dir),
        target=args.target,
        task=args.task,
        optimize_threshold=bool(args.optimize_threshold),
        metric=str(args.metric),
        min_precision=args.min_precision,
        domain_col=str(args.domain_col),
        inject_missing=bool(args.inject_missing),
        missing_row_rate=float(args.missing_row_rate),
        missing_col_rate=float(args.missing_col_rate),
        missing_seed=int(args.missing_seed),
        age_weight=float(args.age_weight),
    )
=======
    train_and_validate(cfg, Path(args.data_dir), target=args.target, task=args.task)
>>>>>>> 56aae0ceea2bf5a2765e2543d64f0e22b032b50e
