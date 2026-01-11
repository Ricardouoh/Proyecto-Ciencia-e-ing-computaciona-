from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.preprocess import infer_columns, load_preprocessor, transform_with_loaded

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


def _load_threshold(threshold_path: Path, fallback: float) -> float:
    if not threshold_path.exists():
        return fallback
    try:
        data = json.loads(threshold_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("threshold") is not None:
            return float(data["threshold"])
    except Exception:
        pass
    return fallback


def _pick_id_column(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for col in ["patient_id", "case_id", "case_submitter_id", "row_id", "id"]:
        if col in df.columns:
            return col
    return None


def _normalize_inputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sex" not in out.columns and "gender" in out.columns:
        out["sex"] = out["gender"]
    if "ethnicity" in out.columns:
        s = out["ethnicity"].astype(str).str.strip().str.lower()
        s = s.replace(
            {
                "not hispanic": "not hispanic or latino",
                "hispanic": "hispanic or latino",
            }
        )
        out["ethnicity"] = s
    if "race" in out.columns:
        s = out["race"].astype(str).str.strip().str.lower()
        s = s.replace(
            {
                "black": "black or african american",
            }
        )
        out["race"] = s
    if "sex" in out.columns:
        out["sex"] = out["sex"].astype(str).str.strip().str.lower()
    for col in ["age_years", "height_last", "weight_last", "bmi_last"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prepare_features(
    df: pd.DataFrame,
    pre,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    X = df.copy()
    for col in ["label", "domain"]:
        if col in X.columns:
            X = X.drop(columns=[col])

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

    Xt = transform_with_loaded(pre, X, numeric_cols, categorical_cols)
    return Xt, missing


def _unwrap_for_shap(model):
    def _unwrap_frozen(obj):
        if obj is None:
            return obj
        if obj.__class__.__name__ != "FrozenEstimator":
            return obj
        for attr in ("estimator_", "estimator"):
            if hasattr(obj, attr):
                inner = getattr(obj, attr)
                if inner is not None:
                    return inner
        return obj

    base = model
    if hasattr(model, "calibrated_classifiers_"):
        cc = model.calibrated_classifiers_[0]
        for attr in ("estimator", "estimator_", "base_estimator", "classifier"):
            if hasattr(cc, attr):
                base = getattr(cc, attr)
                break
    elif hasattr(model, "base_estimator_"):
        base = model.base_estimator_

    seen = set()
    while base is not None and id(base) not in seen:
        seen.add(id(base))
        inner = _unwrap_frozen(base)
        if inner is base:
            break
        base = inner
    return base


def _base_feature(name: str) -> str:
    if name.startswith("num__"):
        return name[len("num__") :]
    if name.startswith("cat__"):
        rest = name[len("cat__") :]
        parts = rest.split("_", 1)
        return parts[0]
    return name


def _aggregate_shap(shap_row: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for val, name in zip(shap_row, feature_names):
        base = _base_feature(str(name))
        agg[base] = agg.get(base, 0.0) + float(val)
    return agg


def _top_pos_neg(agg: Dict[str, float], k: int = 2) -> Tuple[List[str], List[str]]:
    items = list(agg.items())
    pos = [x for x in items if x[1] > 0]
    neg = [x for x in items if x[1] < 0]
    pos_sorted = sorted(pos, key=lambda x: x[1], reverse=True)[:k]
    neg_sorted = sorted(neg, key=lambda x: x[1])[:k]
    pos_vars = [f"{n} ({v:.4f})" for n, v in pos_sorted]
    neg_vars = [f"{n} ({v:.4f})" for n, v in neg_sorted]
    return pos_vars, neg_vars


def _plot_top_contribs(
    agg: Dict[str, float],
    out_path: Path,
    top_k: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    items = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    if not items:
        return
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ["#c44e52" if v < 0 else "#4c72b0" for v in values]
    y = np.arange(len(labels))[::-1]
    plt.figure(figsize=(6.5, 4))
    plt.barh(y, values[::-1], color=colors[::-1])
    plt.yticks(y, labels[::-1])
    plt.axvline(0, color="gray", linewidth=1)
    plt.xlabel("SHAP contribution")
    plt.title("Top feature contributions (highest risk)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Inferencia clinica con explicabilidad.")
    ap.add_argument("--input", required=True, help="CSV de entrada.")
    ap.add_argument("--output", default=None, help="CSV de salida.")
    ap.add_argument("--model", default="results/model.joblib", help="Ruta al modelo.")
    ap.add_argument("--preprocessor", default="results/preprocessor.joblib", help="Ruta al preprocesador.")
    ap.add_argument("--schema", default=None, help="Ruta opcional a schema.json.")
    ap.add_argument("--threshold-json", default="results/threshold.json", help="Ruta a threshold.json.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Umbral manual (si no hay threshold.json).")
    ap.add_argument("--id-col", default=None, help="Columna ID para el resumen.")
    ap.add_argument("--print-limit", type=int, default=0, help="Limite de filas a imprimir (0=todo).")
    ap.add_argument("--explain", action="store_true", help="Activa explicaciones SHAP.")
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

    threshold = _load_threshold(Path(args.threshold_json), float(args.threshold))

    df = pd.read_csv(input_path)
    df = _normalize_inputs(df)
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
    if missing_cols:
        print("WARN Columnas faltantes imputadas:", ", ".join(missing_cols))
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    out_df = df.copy()
    out_df["risk_probability"] = proba
    out_df["prediction"] = np.where(preds == 1, "Riesgo Alto", "Riesgo Bajo")

    out_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_preds.csv")
    out_df.to_csv(out_path, index=False)

    id_col = _pick_id_column(df, args.id_col)
    to_print = len(out_df) if args.print_limit <= 0 else min(len(out_df), args.print_limit)

    shap_rows = None
    shap_feature_names: List[str] = []
    if args.explain:
        if shap is None:
            print("WARN shap no esta instalado. Ejecuta: pip install shap")
        else:
            explain_model = _unwrap_for_shap(model)
            shap_feature_names = list(X.columns)
            try:
                explainer = shap.TreeExplainer(explain_model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_rows = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
                else:
                    shap_rows = np.asarray(shap_values)
                print("INFO SHAP usando modelo base (pre-calibracion).")
            except Exception as exc:
                print(f"WARN SHAP fallo: {exc}")
                shap_rows = None

    for i in range(to_print):
        row = out_df.iloc[i]
        pid = row[id_col] if id_col else i + 1
        pct = float(row["risk_probability"]) * 100.0
        clas = row["prediction"]
        print(f"Paciente ID: {pid} | Probabilidad: {pct:.2f}% | Clasificacion: {clas}")
        if shap_rows is not None and preds[i] == 1:
            agg = _aggregate_shap(shap_rows[i], shap_feature_names)
            pos, neg = _top_pos_neg(agg, k=2)
            if pos:
                print(f"Factores que aumentan el riesgo: {', '.join(pos)}")
            if neg:
                print(f"Factores que protegen o bajan el riesgo: {', '.join(neg)}")

    if shap_rows is not None:
        top_idx = int(np.argmax(proba))
        agg = _aggregate_shap(shap_rows[top_idx], shap_feature_names)
        _plot_top_contribs(agg, out_path.with_name("explanation.png"))

    print(f"OK -> {out_path} ({len(out_df)} filas)")
    print(f"Threshold usado: {threshold}")


if __name__ == "__main__":
    main()
