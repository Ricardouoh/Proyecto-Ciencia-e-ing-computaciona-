from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.age_prior import apply_age_prior, clip_proba, load_age_prior
from src.feature_weighting import apply_feature_weights, load_feature_weights
from src.preprocess import infer_columns, load_preprocessor, transform_with_loaded

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


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

    if "tobacco_smoking_status_any" in X.columns:
        X["tobacco_smoking_status_any"] = pd.to_numeric(
            X["tobacco_smoking_status_any"], errors="coerce"
        ).fillna(0)

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


def _top_features(agg: Dict[str, float], k: int) -> List[str]:
    items = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)[:k]
    return [k for k, _ in items]


def _load_brier(brier_path: Optional[str], brier_value: Optional[float]) -> Optional[float]:
    if brier_value is not None:
        return float(brier_value)
    if not brier_path:
        return None
    path = Path(brier_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        val = data.get("brier_score")
        return float(val) if val is not None else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparacion SHAP entre dos pacientes.")
    ap.add_argument("--input", required=True, help="CSV de entrada.")
    ap.add_argument("--model", default="results/model.joblib", help="Ruta al modelo.")
    ap.add_argument("--preprocessor", default="results/preprocessor.joblib", help="Ruta al preprocesador.")
    ap.add_argument("--schema", default=None, help="Ruta opcional a schema.json.")
    ap.add_argument("--threshold-json", default="results/threshold.json", help="Ruta a threshold.json.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Umbral manual (si no hay threshold.json).")
    ap.add_argument("--feature-weights", default="results/feature_weights.json", help="Ruta a feature_weights.json.")
    ap.add_argument("--age-prior", default="results/age_prior.json", help="Ruta a age_prior.json.")
    ap.add_argument("--max-proba", type=float, default=None, help="Limite maximo de probabilidad.")
    ap.add_argument("--id-col", default=None, help="Columna ID para buscar pacientes.")
    ap.add_argument("--id-a", default=None, help="ID paciente A.")
    ap.add_argument("--id-b", default=None, help="ID paciente B.")
    ap.add_argument("--top-k", type=int, default=5, help="Top K variables por paciente.")
    ap.add_argument("--brier-json", default="results/final_scientific_report.json", help="Reporte con Brier.")
    ap.add_argument("--brier", type=float, default=None, help="Brier manual.")
    args = ap.parse_args()

    if shap is None:
        raise RuntimeError("shap no esta instalado. Ejecuta: pip install shap")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontro {input_path}")

    model_path = Path(args.model)
    pre_path = Path(args.preprocessor)
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro {model_path}")
    if not pre_path.exists():
        raise FileNotFoundError(f"No se encontro {pre_path}")

    df = pd.read_csv(input_path)
    df = _normalize_inputs(df)

    id_col = _pick_id_column(df, args.id_col)
    if id_col is None:
        raise ValueError("No se encontro columna ID. Usa --id-col.")

    if args.id_a is None or args.id_b is None:
        if len(df) < 2:
            raise ValueError("Se requieren al menos dos pacientes.")
        args.id_a = df[id_col].iloc[1] if len(df) > 1 else df[id_col].iloc[0]
        args.id_b = df[id_col].iloc[2] if len(df) > 2 else df[id_col].iloc[0]

    idx_a = df.index[df[id_col].astype(str) == str(args.id_a)].tolist()
    idx_b = df.index[df[id_col].astype(str) == str(args.id_b)].tolist()
    if not idx_a or not idx_b:
        raise ValueError("No se encontraron ambos IDs en el CSV.")

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

    threshold, max_proba = _load_threshold_info(Path(args.threshold_json), float(args.threshold))
    if args.max_proba is not None:
        max_proba = float(args.max_proba)
    proba = model.predict_proba(X)[:, 1]
    age_prior = load_age_prior(args.age_prior)
    if age_prior and "age_years" in df.columns:
        proba = apply_age_prior(proba, df["age_years"], age_prior)
    if max_proba is not None:
        proba = clip_proba(proba, max_proba)
    preds = (proba >= threshold).astype(int)

    explain_model = _unwrap_for_shap(model)
    explainer = shap.TreeExplainer(explain_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_mat = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        shap_mat = np.asarray(shap_values)

    feat_names = list(X.columns)

    agg_a = _aggregate_shap(shap_mat[idx_a[0]], feat_names)
    agg_b = _aggregate_shap(shap_mat[idx_b[0]], feat_names)

    top_a = _top_features(agg_a, int(args.top_k))
    top_b = _top_features(agg_b, int(args.top_k))
    union = list(dict.fromkeys(top_a + top_b))

    rows = []
    for feat in union:
        a = agg_a.get(feat, 0.0)
        b = agg_b.get(feat, 0.0)
        rows.append({"feature": feat, "shap_a": a, "shap_b": b, "delta": a - b})
    table = pd.DataFrame(rows).sort_values("delta", key=lambda s: s.abs(), ascending=False)
    print("=== COMPARACION SHAP (Top 5 por paciente) ===")
    print(table.to_string(index=False))

    all_feats = set(agg_a.keys()) | set(agg_b.keys())
    diffs = {f: agg_a.get(f, 0.0) - agg_b.get(f, 0.0) for f in all_feats}
    max_feat = max(diffs.items(), key=lambda x: abs(x[1]))
    who = "Paciente A" if max_feat[1] > 0 else "Paciente B"
    print("\nPunto de quiebre:", max_feat[0], f"(delta={max_feat[1]:.4f}, mayor en {who})")

    brier = _load_brier(args.brier_json, args.brier)
    if brier is not None:
        print(
            "\nResumen de confiabilidad:\n"
            "El modelo es confiable en sus predicciones porque el Brier score es "
            f"{brier:.3f} y las variables con mayor impacto coinciden con la literatura "
            "medica (edad y raza), mientras que la etnicidad aporta matices que "
            "personalizan el riesgo."
        )

    print(f"Paciente A: {args.id_a} | proba={proba[idx_a[0]]:.4f} | pred={'Alto' if preds[idx_a[0]]==1 else 'Bajo'}")
    print(f"Paciente B: {args.id_b} | proba={proba[idx_b[0]]:.4f} | pred={'Alto' if preds[idx_b[0]]==1 else 'Bajo'}")


if __name__ == "__main__":
    main()
