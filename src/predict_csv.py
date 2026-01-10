from __future__ import annotations
"""
Corre inferencia sobre un CSV con las columnas crudas usadas por el modelo.

Uso:
  python -m src.predict_csv --input data/docs/model_input_template.csv

Salida:
  CSV con columnas originales + proba_cancer + pred_label.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Needed for loading preprocessors pickled when AgeWeightAdder existed in __main__.
from src.feature_transforms import AgeWeightAdder  # noqa: F401


DEFAULT_BUNDLE = "results_calibrated_reweighted/model_bundle.joblib"


def _resolve_model(bundle_path: str, model_path: str | None, pre_path: str | None):
    """
    Carga un bundle calibrado o un modelo + preprocesador separados.
    Retorna (preprocessor, model).
    """
    if model_path and pre_path:
        pre = joblib.load(pre_path)
        model = joblib.load(model_path)
        return pre, model

    if bundle_path and str(bundle_path).strip().lower() not in {"", "none", "null", "skip"}:
        bundle = joblib.load(bundle_path)
        if isinstance(bundle, dict) and "preprocessor" in bundle and "calibrated" in bundle:
            return bundle["preprocessor"], bundle["calibrated"]

    raise ValueError("No se pudo cargar el modelo. Usa --bundle o --model + --preprocessor.")


def _required_columns(preprocessor) -> List[str]:
    """
    Obtiene columnas crudas esperadas por el preprocesador.
    """
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    # Fallback: columnas conocidas del proyecto
    return [
        "age_years",
        "sex",
        "ethnicity",
        "race",
        "height_last",
        "weight_last",
        "bmi_last",
        "tobacco_smoking_status_any",
    ]


def _prepare_features(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    """
    Asegura columnas requeridas y retorna en el orden correcto.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Create missing columns as NaN to allow incomplete clinical records.
        for col in missing:
            df[col] = np.nan
    return df[required]


def _normalize_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza valores comunes en entradas crudas (strings -> categorias/nums).
    """
    df2 = df.copy()

    # Normalize categorical fields
    for col in ["sex", "ethnicity", "race"]:
        if col not in df2.columns:
            continue
        s = df2[col].astype(str).str.strip()
        s_lower = s.str.lower()
        invalid = s_lower.isin(["", "nan", "none", "not reported", "unknown"])
        s_clean = s_lower.where(~invalid, pd.NA)
        if col == "race":
            s_clean = s_clean.replace({"unknown": "Unknown"})
        df2[col] = s_clean

    # Tobacco mapping (accept strings like current/never/former)
    if "tobacco_smoking_status_any" in df2.columns:
        s = df2["tobacco_smoking_status_any"]
        if pd.api.types.is_numeric_dtype(s):
            df2["tobacco_smoking_status_any"] = pd.to_numeric(s, errors="coerce")
        else:
            s_str = s.astype(str).str.lower().str.strip()
            invalid = s_str.isin(["", "nan", "none"])
            s_str = s_str.where(~invalid, pd.NA)
            mapping = {
                "current": 1,
                "former": 1,
                "ever": 1,
                "smoker": 1,
                "yes": 1,
                "y": 1,
                "true": 1,
                "1": 1,
                "never": 0,
                "no": 0,
                "n": 0,
                "false": 0,
                "0": 0,
            }
            mapped = s_str.map(mapping)
            df2["tobacco_smoking_status_any"] = pd.to_numeric(mapped, errors="coerce")

    # Numeric fields
    for col in ["age_years", "height_last", "weight_last", "bmi_last"]:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")

    # Ensure pandas NA values become np.nan for sklearn imputers.
    df2 = df2.where(pd.notna(df2), np.nan)
    return df2


def run_inference(
    input_path: str,
    output_path: str | None,
    bundle_path: str,
    model_path: str | None,
    pre_path: str | None,
    threshold: float,
    labels_path: str | None,
    metric: str,
    min_precision: float | None,
    optimize_threshold: bool,
) -> Tuple[Path, int, int]:
    df = pd.read_csv(input_path)
    df = _normalize_inputs(df)

    preprocessor, model = _resolve_model(bundle_path, model_path, pre_path)
    required = _required_columns(preprocessor)

    features = df.drop(columns=["label"], errors="ignore")
    X_raw = _prepare_features(features, required)
    X = preprocessor.transform(X_raw)
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        if X.shape[1] != len(feature_names):
            raise ValueError(
                "Mismatch entre columnas transformadas y el modelo. "
                f"X tiene {X.shape[1]} columnas, modelo espera {len(feature_names)}."
            )
        X = pd.DataFrame(X, columns=feature_names)
    elif hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
        if X.shape[1] == len(feature_names):
            X = pd.DataFrame(X, columns=feature_names)

    proba = model.predict_proba(X)[:, 1]

    eval_rows = None
    best_threshold = None
    best_metrics = None
    if labels_path:
        labels = pd.read_csv(labels_path)
        if "true_label" not in labels.columns:
            raise ValueError("El archivo de labels debe tener columna true_label.")
        if "row_id" in labels.columns and "row_id" in df.columns:
            label_map = dict(zip(labels["row_id"], labels["true_label"]))
            y_true = df["row_id"].map(label_map)
        else:
            if len(labels) != len(df):
                raise ValueError("Labels sin row_id requieren el mismo numero de filas.")
            y_true = labels["true_label"]

        mask = y_true.notna()
        y_true = y_true[mask].astype(int)
        proba_eval = proba[mask.to_numpy()]

        thresholds = [i / 100 for i in range(0, 101)]
        eval_rows = []
        for th in thresholds:
            y_pred = (proba_eval >= th).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            eval_rows.append(
                {
                    "threshold": th,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "pred_pos": int((y_pred == 1).sum()),
                }
            )

        best_metrics = eval_rows[0] if eval_rows else None
        best_threshold = best_metrics["threshold"] if best_metrics else threshold
        for row in eval_rows:
            if metric == "recall" and min_precision is not None:
                if row["precision"] < min_precision:
                    continue
            score = row.get(metric)
            if best_metrics is None or score > best_metrics.get(metric, -1):
                best_metrics = row
                best_threshold = row["threshold"]
            elif best_metrics is not None and score == best_metrics.get(metric):
                # Tie-breaker: prefer higher precision when optimizing recall.
                if metric == "recall" and row.get("precision", 0) > best_metrics.get("precision", 0):
                    best_metrics = row
                    best_threshold = row["threshold"]

        if optimize_threshold and best_threshold is not None:
            threshold = best_threshold

    preds = (proba >= threshold).astype(int)

    out_df = df.copy()
    out_df["proba_cancer"] = proba
    out_df["pred_label"] = preds

    out_path = Path(output_path) if output_path else Path(input_path).with_name(
        f"{Path(input_path).stem}_preds.csv"
    )
    out_df.to_csv(out_path, index=False)

    if labels_path and eval_rows is not None:
        eval_df = pd.DataFrame(eval_rows)
        eval_path = out_path.with_name(f"{out_path.stem}_threshold_scan.csv")
        eval_df.to_csv(eval_path, index=False)

        used_row = None
        if eval_rows:
            used_row = min(eval_rows, key=lambda r: abs(r["threshold"] - threshold))

        summary = {
            "threshold_used": threshold,
            "metrics_at_threshold": used_row,
            "metric": metric,
            "best_threshold": best_threshold,
            "best_metrics": best_metrics,
        }
        summary_path = out_path.with_name(f"{out_path.stem}_eval.json")
        pd.Series(summary).to_json(summary_path, indent=2)

    positives = int((preds == 1).sum())
    negatives = int((preds == 0).sum())
    return out_path, positives, negatives


def main() -> None:
    ap = argparse.ArgumentParser(description="Inferencia de cancer (label=1) desde CSV.")
    ap.add_argument("--input", required=True, help="Ruta al CSV de entrada.")
    ap.add_argument(
        "--output",
        default=None,
        help="Ruta al CSV de salida. Default: <input>_preds.csv",
    )
    ap.add_argument(
        "--bundle",
        default=DEFAULT_BUNDLE,
        help="Ruta al bundle calibrado (preprocessor + calibrated). Se ignora si usas --model y --preprocessor.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Ruta al modelo si no usas bundle (opcional).",
    )
    ap.add_argument(
        "--preprocessor",
        default=None,
        help="Ruta al preprocesador si no usas bundle (opcional).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral para clasificar label=1.",
    )
    ap.add_argument(
        "--labels",
        default=None,
        help="CSV con true_label (y row_id opcional) para evaluar/optimizar umbral.",
    )
    ap.add_argument(
        "--metric",
        default="f1",
        choices=["f1", "recall", "precision", "accuracy"],
        help="Metrica para elegir umbral cuando hay labels.",
    )
    ap.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Precision minima si metric=recall (opcional).",
    )
    ap.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Usa el mejor umbral segun --metric cuando hay labels.",
    )

    args = ap.parse_args()
    out_path, positives, negatives = run_inference(
        input_path=args.input,
        output_path=args.output,
        bundle_path=args.bundle,
        model_path=args.model,
        pre_path=args.preprocessor,
        threshold=args.threshold,
        labels_path=args.labels,
        metric=args.metric,
        min_precision=args.min_precision,
        optimize_threshold=args.optimize_threshold,
    )

    total = positives + negatives
    print(f"OK -> {out_path} ({total} filas)")
    print(f"Pred label=1: {positives} | label=0: {negatives}")


if __name__ == "__main__":
    main()
