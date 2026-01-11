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
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - compat fallback
    FrozenEstimator = None
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


def _get_model_class_weight(model) -> Optional[object]:
    try:
        params = model.get_params()
        return params.get("class_weight", None)
    except Exception:
        return None


def _combine_sample_weights(
    base_weights: Optional[np.ndarray],
    extra_weights: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if base_weights is None and extra_weights is None:
        return None
    if base_weights is None:
        return extra_weights
    if extra_weights is None:
        return base_weights
    if len(base_weights) != len(extra_weights):
        return base_weights
    return base_weights * extra_weights


def _duplicate_weights(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if weights is None or mask is None:
        return weights
    if len(weights) != len(mask):
        return weights
    return np.concatenate([weights, weights[mask]])


def _apply_pu_label_smoothing(
    X: pd.DataFrame,
    y: pd.Series,
    domain: Optional[pd.Series],
    unlabeled_domain: str,
    unlabeled_prob: float,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], Optional[np.ndarray], Optional[np.ndarray]]:
    if unlabeled_prob <= 0.0 or domain is None:
        return X, y, domain, None, None
    mask = (domain.astype(str) == str(unlabeled_domain)) & (y.astype(int) == 0)
    if mask.sum() == 0:
        return X, y, domain, None, None

    X_dup = X.loc[mask].copy()
    y_dup = pd.Series(np.ones(int(mask.sum()), dtype=int))
    domain_dup = domain.loc[mask].copy()

    X_new = pd.concat([X, X_dup], ignore_index=True)
    y_new = pd.concat([y, y_dup], ignore_index=True)
    domain_new = pd.concat([domain, domain_dup], ignore_index=True)

    pu_weights = np.ones(len(y), dtype=float)
    pu_weights[mask.to_numpy()] = 1.0 - float(unlabeled_prob)
    pu_weights = np.concatenate([pu_weights, np.full(int(mask.sum()), float(unlabeled_prob))])
    return X_new, y_new, domain_new, pu_weights, mask.to_numpy()


def _fit_with_sample_weight(model, X, y, sample_weight):
    if sample_weight is None:
        model.fit(X, y)
        return
    try:
        sig = inspect.signature(model.fit)
        if "sample_weight" in sig.parameters:
            model.fit(X, y, sample_weight=sample_weight)
            return
    except (TypeError, ValueError):
        pass
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)


def _compute_risk_weights(
    X: pd.DataFrame,
    min_age: float,
    weight_factor: float,
    use_race: bool,
) -> np.ndarray:
    """
    Compute light risk weights based on age, smoker, and optionally race.
    Weights are in [1, 1 + weight_factor].
    """
    n = len(X)
    if n == 0:
        return np.array([])
    risk = np.zeros(n, dtype=float)
    max_score = 0.0

    if "age_years" in X.columns:
        age = pd.to_numeric(X["age_years"], errors="coerce")
        risk += (age >= float(min_age)).fillna(False).astype(float)
        max_score += 1.0

    if "tobacco_smoking_status_any" in X.columns:
        smoker = pd.to_numeric(X["tobacco_smoking_status_any"], errors="coerce")
        risk += (smoker == 1).fillna(False).astype(float)
        max_score += 1.0

    if use_race and "race" in X.columns:
        race = X["race"].astype(str).str.strip().str.lower()
        risk += (race == "white").fillna(False).astype(float)
        max_score += 1.0

    if max_score <= 0:
        return np.ones(n, dtype=float)
    score = risk / max_score
    return 1.0 + float(weight_factor) * score


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


def _domain_pos_rate_gap(
    domains: pd.Series,
    y_pred: np.ndarray,
) -> Tuple[float, float, float]:
    if len(domains) != len(y_pred):
        return 0.0, 0.0, 0.0
    rates: list[float] = []
    for dom in sorted(domains.dropna().unique()):
        mask = domains == dom
        if mask.sum() == 0:
            continue
        rates.append(float(np.mean(y_pred[mask.to_numpy()])))
    if not rates:
        return 0.0, 0.0, 0.0
    min_rate = float(min(rates))
    max_rate = float(max(rates))
    gap = max_rate - min_rate if len(rates) > 1 else 0.0
    return gap, min_rate, max_rate


def _scan_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    domains: Optional[pd.Series] = None,
    target_domain: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for t in np.linspace(0.0, 1.0, 101):
        metrics = mt.compute_classification_metrics(y_true, proba, threshold=float(t))
        metrics["threshold"] = float(t)
        metrics["pred_pos"] = int((proba >= t).sum())
        if domains is not None:
            y_pred = (proba >= t).astype(int)
            gap, min_rate, max_rate = _domain_pos_rate_gap(domains, y_pred)
            metrics["domain_pos_rate_gap"] = gap
            metrics["domain_pos_rate_min"] = min_rate
            metrics["domain_pos_rate_max"] = max_rate
            if target_domain is not None:
                mask = domains.astype(str) == str(target_domain)
                if mask.sum() > 0:
                    metrics["target_domain_pos_rate"] = float(np.mean(y_pred[mask.to_numpy()]))
                else:
                    metrics["target_domain_pos_rate"] = float("nan")
        rows.append(metrics)
    return pd.DataFrame(rows)


def _choose_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric: str,
    min_precision: Optional[float],
    domains: Optional[pd.Series] = None,
    domain_penalty_weight: float = 0.0,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    min_recall: float = 0.0,
    target_domain: Optional[str] = None,
    target_pos_rate: Optional[float] = None,
    target_weight: float = 0.0,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    scan = _scan_thresholds(y_true, proba, domains=domains, target_domain=target_domain)
    best = None
    best_score = None
    filtered = []
    for _, row in scan.iterrows():
        if row["threshold"] < float(min_threshold) or row["threshold"] > float(max_threshold):
            continue
        if min_recall > 0.0 and row.get("recall", 0.0) < min_recall:
            continue
        filtered.append(row)

    if not filtered:
        filtered = []
        for _, row in scan.iterrows():
            if row["threshold"] < float(min_threshold) or row["threshold"] > float(max_threshold):
                continue
            filtered.append(row)

    for row in filtered:
        if metric == "recall" and min_precision is not None and row["precision"] < min_precision:
            continue
        score = row[metric]
        if domain_penalty_weight > 0 and "domain_pos_rate_gap" in row:
            score = score - domain_penalty_weight * row["domain_pos_rate_gap"]
        if (
            target_weight > 0
            and target_pos_rate is not None
            and "target_domain_pos_rate" in row
            and np.isfinite(row["target_domain_pos_rate"])
        ):
            score = score - target_weight * abs(row["target_domain_pos_rate"] - float(target_pos_rate))
        if best is None or best_score is None or score > best_score:
            best = row
            best_score = score
        elif best is not None and best_score is not None and score == best_score:
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
    optimize_threshold: bool = False,
    metric: str = "f1",
    min_precision: Optional[float] = None,
    domain_col: str = "domain",
    inject_missing: bool = False,
    missing_row_rate: float = 0.3,
    missing_col_rate: float = 0.2,
    missing_seed: int = 42,
    age_weight: float = 1.0,
    risk_weight: bool = False,
    risk_weight_factor: float = 0.5,
    risk_weight_min_age: float = 60.0,
    risk_weight_use_race: bool = False,
    domain_penalty_weight: float = 0.0,
    min_threshold: float = 0.10,
    max_threshold: float = 0.50,
    min_recall: float = 0.80,
    pu_label_smoothing: float = 0.0,
    pu_unlabeled_domain: str = "nhanes",
    pu_target_pos_rate: Optional[float] = None,
    pu_target_weight: float = 0.0,
    calibrate: bool = False,
    calibration_method: str = "sigmoid",
) -> None:
    if min_threshold > max_threshold:
        raise ValueError("min_threshold no puede ser mayor que max_threshold.")
    # Cargar datos (prioriza aligned_* si existen)
    train_path, val_path, test_path = _resolve_split_paths(data_dir)
    Xtr, ytr, domain_tr = _load_xy(train_path, target=target, task=task, domain_col=domain_col)
    Xva, yva, domain_va = _load_xy(val_path, target=target, task=task, domain_col=domain_col)
    Xte, yte, domain_te = _load_xy(test_path, target=target, task=task, domain_col=domain_col)

    Xtr_raw = Xtr.copy()
    risk_weights = None
    if risk_weight:
        risk_weights = _compute_risk_weights(
            Xtr_raw,
            min_age=risk_weight_min_age,
            weight_factor=risk_weight_factor,
            use_race=risk_weight_use_race,
        )
        if len(risk_weights) == len(Xtr_raw):
            print(
                "INFO Risk weights enabled:"
                f" min_age={risk_weight_min_age}"
                f" factor={risk_weight_factor}"
                f" use_race={risk_weight_use_race}"
            )
            print(
                "INFO Risk weights stats:"
                f" min={risk_weights.min():.3f}"
                f" max={risk_weights.max():.3f}"
                f" mean={risk_weights.mean():.3f}"
            )
        else:
            print("WARN Risk weights size mismatch; disabling risk weights.")
            risk_weights = None

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

    pu_weights = None
    pu_mask = None
    if task == "classification":
        Xtr, ytr, domain_tr, pu_weights, pu_mask = _apply_pu_label_smoothing(
            Xtr,
            ytr,
            domain_tr,
            unlabeled_domain=pu_unlabeled_domain,
            unlabeled_prob=float(pu_label_smoothing),
        )
        if risk_weights is not None and pu_mask is not None:
            risk_weights = _duplicate_weights(risk_weights, pu_mask)

    # Preprocesador (imputacion + escala + one-hot)
    pre_path = cfg.outdir / "preprocessor.joblib"
    Xtr, Xva, Xte = _fit_preprocessor(
        Xtr, Xva, Xte,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        out_path=pre_path,
        age_weight=age_weight,
    )

    # Construir modelo
    model = make_model(cfg.model_cfg, task=task)

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

    # ===== Clasificacion =====
    is_mlp = model.__class__.__name__.lower().startswith("mlp")
    if not is_mlp:
        base_sw = None
        if _get_model_class_weight(model) is None and pu_label_smoothing <= 0.0:
            base_sw = _class_weight_sample_weights(ytr)
        sw = _combine_sample_weights(base_sw, risk_weights)
        sw = _combine_sample_weights(sw, pu_weights)
        _fit_with_sample_weight(model, Xtr, ytr, sw)
        final_model = model
        if calibrate:
            try:
                base = final_model
                if FrozenEstimator is not None:
                    base = FrozenEstimator(final_model)
                calib = CalibratedClassifierCV(base, method=str(calibration_method))
                calib.fit(Xva, yva)
                final_model = calib
            except Exception as exc:
                print(f"WARN Calibration failed: {exc}")

        proba_va = final_model.predict_proba(Xva)[:, 1]

        metrics = _eval_metrics(yva.values, proba_va)
        log_rows.append({"epoch": 1, **metrics})

        _save_log(log_rows, log_path)
        _save_model(final_model, model_path)

        full_metrics = mt.compute_classification_metrics(yva.values, proba_va)
        mt.save_json(full_metrics, cfg.outdir / "metrics_val.json")
        mt.plot_roc(yva.values, proba_va, cfg.outdir / "roc_val.png")
        mt.plot_pr(yva.values, proba_va, cfg.outdir / "pr_val.png")
        preds_va = (proba_va >= 0.5).astype(int)
        mt.plot_confusion_matrix(yva.values, preds_va, cfg.outdir / "confusion_matrix_val.png")

        print("OK Modelo guardado:", model_path)
        print(f"OK Metricas validacion: AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f}")
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

        base_sw = None
        if pu_label_smoothing <= 0.0:
            base_sw = _class_weight_sample_weights(ytr)
        sw = _combine_sample_weights(base_sw, risk_weights)
        sw = _combine_sample_weights(sw, pu_weights)

        best_auroc = -np.inf
        best_model = None
        no_improve = 0

        for epoch in range(1, cfg.max_epochs + 1):
            _fit_with_sample_weight(model, Xtr, ytr, sw)

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

            if no_improve >= cfg.patience:
                print("STOP Early stopping activado")
                break

        _save_log(log_rows, log_path)

        final_model = best_model if best_model is not None else model
        if calibrate:
            try:
                base = final_model
                if FrozenEstimator is not None:
                    base = FrozenEstimator(final_model)
                calib = CalibratedClassifierCV(base, method=str(calibration_method))
                calib.fit(Xva, yva)
                final_model = calib
            except Exception as exc:
                print(f"WARN Calibration failed: {exc}")
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
        print(
            f"OK Metricas validacion (mejor modelo):"
            f" AUROC={final_metrics['auroc']:.4f}"
            f" | AUPRC={final_metrics['auprc']:.4f}"
        )

    # Seleccion de umbral
    threshold = 0.5
    best_val = None
    target_domain = pu_unlabeled_domain if pu_target_pos_rate is not None else None
    scan = _scan_thresholds(yva.values, proba_va, domains=domain_va, target_domain=target_domain)
    scan.to_csv(cfg.outdir / "threshold_scan.csv", index=False)
    if optimize_threshold:
        threshold, best_val, _ = _choose_threshold(
            yva.values,
            proba_va,
            metric=metric,
            min_precision=min_precision,
            domains=domain_va,
            domain_penalty_weight=domain_penalty_weight,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            min_recall=min_recall,
            target_domain=target_domain,
            target_pos_rate=pu_target_pos_rate,
            target_weight=pu_target_weight,
        )
    threshold_info = {
        "threshold": threshold,
        "metric": metric,
        "min_precision": min_precision,
        "best_val": best_val,
        "domain_penalty_weight": domain_penalty_weight,
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "min_recall": min_recall,
        "pu_label_smoothing": pu_label_smoothing,
        "pu_unlabeled_domain": pu_unlabeled_domain,
        "pu_target_pos_rate": pu_target_pos_rate,
        "pu_target_weight": pu_target_weight,
        "calibrated": bool(calibrate),
        "calibration_method": calibration_method if calibrate else None,
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
    ap.add_argument("--optimize-threshold", action="store_true", help="Optimiza umbral en validacion.")
    ap.add_argument(
        "--metric",
        default="f1",
        choices=["recall", "f1", "precision", "accuracy"],
        help="Metrica para seleccionar umbral (default=f1).",
    )
    ap.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Precision minima si metric=recall (opcional).",
    )
    ap.add_argument(
        "--min-threshold",
        type=float,
        default=0.10,
        help="Umbral minimo permitido al optimizar (default=0.10).",
    )
    ap.add_argument(
        "--max-threshold",
        type=float,
        default=0.50,
        help="Umbral maximo permitido al optimizar (default=0.50).",
    )
    ap.add_argument(
        "--min-recall",
        type=float,
        default=0.80,
        help="Recall minimo permitido al optimizar (default=0.80).",
    )
    ap.add_argument("--inject-missing", action="store_true", help="Inyecta missingness en train.")
    ap.add_argument("--missing-row-rate", type=float, default=0.3, help="Prob de filas con missing.")
    ap.add_argument("--missing-col-rate", type=float, default=0.2, help="Prob de NaN por columna.")
    ap.add_argument("--missing-seed", type=int, default=42, help="Seed para missingness.")
    ap.add_argument("--domain-col", default="domain", help="Columna de dominio para metricas.")
    ap.add_argument("--age-weight", type=float, default=1.0, help="Peso extra para age_years.")
    ap.add_argument(
        "--domain-penalty-weight",
        type=float,
        default=0.0,
        help="Penaliza diferencia de tasa de positivos entre dominios al optimizar umbral.",
    )
    ap.add_argument("--risk-weight", action="store_true", help="Activa ponderacion por grupo de riesgo.")
    ap.add_argument(
        "--risk-weight-factor",
        type=float,
        default=0.5,
        help="Factor de peso extra para grupo de riesgo (0-1).",
    )
    ap.add_argument(
        "--risk-weight-min-age",
        type=float,
        default=60.0,
        help="Edad minima para riesgo (default 60).",
    )
    ap.add_argument(
        "--risk-weight-use-race",
        action="store_true",
        help="Incluye race=white en el score de riesgo.",
    )
    ap.add_argument(
        "--pu-label-smoothing",
        type=float,
        default=0.0,
        help="Suaviza etiquetas 0 en dominio no etiquetado (ej 0.05).",
    )
    ap.add_argument(
        "--pu-unlabeled-domain",
        default="nhanes",
        help="Dominio no etiquetado para PU (default=nhanes).",
    )
    ap.add_argument(
        "--pu-target-pos-rate",
        type=float,
        default=None,
        help="Tasa objetivo de positivos en el dominio no etiquetado al optimizar umbral.",
    )
    ap.add_argument(
        "--pu-target-weight",
        type=float,
        default=0.0,
        help="Peso de la penalizacion por desviacion de la tasa objetivo.",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Aplica calibracion probabilistica en validacion.",
    )
    ap.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Metodo de calibracion (sigmoid o isotonic).",
    )
    args = ap.parse_args()

    # Soporta default_config como función o como diccionario
    if callable(default_config):
        model_cfg = default_config(args.model)
    else:
        model_cfg = default_config[args.model]

    if args.task == "regression" and args.model in ("logreg", "mlp"):
        raise ValueError("Modelo no valido para regresion. Usa 'gbr' o 'rf'.")

    cfg = TrainConfig(model_cfg=model_cfg)
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
        risk_weight=bool(args.risk_weight),
        risk_weight_factor=float(args.risk_weight_factor),
        risk_weight_min_age=float(args.risk_weight_min_age),
        risk_weight_use_race=bool(args.risk_weight_use_race),
        domain_penalty_weight=float(args.domain_penalty_weight),
        min_threshold=float(args.min_threshold),
        max_threshold=float(args.max_threshold),
        min_recall=float(args.min_recall),
        pu_label_smoothing=float(args.pu_label_smoothing),
        pu_unlabeled_domain=str(args.pu_unlabeled_domain),
        pu_target_pos_rate=args.pu_target_pos_rate,
        pu_target_weight=float(args.pu_target_weight),
        calibrate=bool(args.calibrate),
        calibration_method=str(args.calibration_method),
    )
