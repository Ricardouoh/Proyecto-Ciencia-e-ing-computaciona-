from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from src.age_prior import apply_age_prior, clip_proba, load_age_prior
from src.feature_weighting import apply_feature_weights, load_feature_weights
from src.preprocess import infer_columns, load_preprocessor, transform_with_loaded


def _load_threshold_info(threshold_path: Path, fallback: float) -> tuple[float, Optional[float]]:
    threshold = fallback
    max_proba: Optional[float] = None
    if not threshold_path.exists():
        return threshold, max_proba
    try:
        data = json.loads(threshold_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("threshold") is not None:
            threshold = float(data["threshold"])
        if isinstance(data, dict) and data.get("max_proba") is not None:
            max_proba = float(data["max_proba"])
    except Exception:
        pass
    return threshold, max_proba


def _pick_id_column(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for col in ["patient_id", "case_id", "case_submitter_id", "row_id", "id"]:
        if col in df.columns:
            return col
    return None


def _prepare_features(
    df: pd.DataFrame,
    pre,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> tuple[pd.DataFrame, List[str]]:
    X = df.copy()
    if "label" in X.columns:
        X = X.drop(columns=["label"])
    if "domain" in X.columns:
        X = X.drop(columns=["domain"])

    missing: List[str] = []
    if hasattr(pre, "feature_names_in_"):
        required = list(pre.feature_names_in_)
        missing = [c for c in required if c not in X.columns]
        for col in missing:
            X[col] = np.nan
        X = X[required]
    else:
        required = list(dict.fromkeys(numeric_cols + categorical_cols))
        missing = [c for c in required if c not in X.columns]
        for col in missing:
            X[col] = np.nan
        if required:
            X = X[required]
    if "tobacco_smoking_status_any" in X.columns:
        X["tobacco_smoking_status_any"] = pd.to_numeric(
            X["tobacco_smoking_status_any"], errors="coerce"
        ).fillna(0)
    Xt = transform_with_loaded(pre, X, numeric_cols, categorical_cols)
    return Xt, missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Inferencia clinica con modelo calibrado.")
    ap.add_argument("--input", required=True, help="CSV de entrada.")
    ap.add_argument("--output", default=None, help="CSV de salida.")
    ap.add_argument("--model", default="results/model.joblib", help="Ruta al modelo.")
    ap.add_argument("--preprocessor", default="results/preprocessor.joblib", help="Ruta al preprocesador.")
    ap.add_argument("--schema", default=None, help="Ruta opcional a schema.json.")
    ap.add_argument("--threshold-json", default="results/threshold.json", help="Ruta a threshold.json.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Umbral manual (si no hay threshold.json).")
    ap.add_argument("--max-proba", type=float, default=None, help="Limite maximo de probabilidad.")
    ap.add_argument("--feature-weights", default="results/feature_weights.json", help="Ruta a feature_weights.json.")
    ap.add_argument("--age-prior", default="results/age_prior.json", help="Ruta a age_prior.json.")
    ap.add_argument("--id-col", default=None, help="Columna ID para el resumen.")
    ap.add_argument("--print-limit", type=int, default=0, help="Limite de filas a imprimir (0=todo).")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontro {input_path}")

    model_path = Path(args.model)
    pre_path = Path(args.preprocessor)
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro {model_path}")
    if not pre_path.exists():
        raise FileNotFoundError(f"No se encontro {pre_path}")

    threshold, max_proba = _load_threshold_info(Path(args.threshold_json), float(args.threshold))
    if args.max_proba is not None:
        max_proba = float(args.max_proba)

    df = pd.read_csv(input_path)
    model = joblib.load(model_path)
    pre = load_preprocessor(pre_path)

    schema_numeric: List[str] = []
    schema_categorical: List[str] = []
    if not hasattr(pre, "feature_names_in_"):
        schema_path = Path(args.schema) if args.schema else None
        if schema_path and schema_path.exists():
            data = json.loads(schema_path.read_text(encoding="utf-8"))
            schema_numeric = [c for c in data.get("numeric", []) if c in df.columns]
            schema_categorical = [c for c in data.get("categorical", []) if c in df.columns]
        if not schema_numeric and not schema_categorical:
            schema_numeric, schema_categorical = infer_columns(df.assign(label=0), target="label")

    X, missing_cols = _prepare_features(df, pre, schema_numeric, schema_categorical)
    weights = load_feature_weights(args.feature_weights)
    if weights:
        X = apply_feature_weights(X, weights)
    if missing_cols:
        print("WARN Columnas faltantes imputadas:", ", ".join(missing_cols))
    proba = model.predict_proba(X)[:, 1]
    age_prior = load_age_prior(args.age_prior)
    if age_prior and "age_years" in df.columns:
        proba = apply_age_prior(proba, df["age_years"], age_prior)
    if max_proba is not None:
        proba = clip_proba(proba, max_proba)
    preds = (proba >= threshold).astype(int)

    out_df = df.copy()
    out_df["risk_probability"] = proba
    out_df["prediction"] = np.where(preds == 1, "Riesgo Alto", "Riesgo Bajo")

    out_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_preds.csv")
    out_df.to_csv(out_path, index=False)

    id_col = _pick_id_column(df, args.id_col)
    to_print = len(out_df) if args.print_limit <= 0 else min(len(out_df), args.print_limit)
    for i in range(to_print):
        row = out_df.iloc[i]
        pid = row[id_col] if id_col else i + 1
        pct = float(row["risk_probability"]) * 100.0
        clas = row["prediction"]
        print(f"Paciente ID: {pid} | Probabilidad: {pct:.2f}% | Clasificacion: {clas}")

    print(f"OK -> {out_path} ({len(out_df)} filas)")
    print(f"Threshold usado: {threshold}")


if __name__ == "__main__":
    main()
