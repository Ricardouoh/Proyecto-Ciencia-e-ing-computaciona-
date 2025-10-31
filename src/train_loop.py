from __future__ import annotations
"""
Entrenamiento supervisado con validación y early stopping.


- Carga train/val desde data/processed (CSV)
- Crea el modelo vía src.model.make_model(cfg)
- LOGREG: fit único + evaluación en val
- MLP: entrenamiento incremental (warm_start) con early stopping (patience=10)
- Guarda:
    - results/model.joblib  (mejor checkpoint por AUROC_val)
    - results/train_log.csv (histórico por época)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

MODEL_PATH = Path("results/model.joblib")
PREPROC_PATH = Path("results/preprocessor.joblib")

app = FastAPI(title="Clinical ML Inference API", version="1.0.0")


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Mapa columna -> valor")


class PredictResponse(BaseModel):
    ok: bool
    proba: float
    label: int
    threshold: float
    missing: List[str] = []
    extra: List[str] = []


def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    if not PREPROC_PATH.exists():
        raise FileNotFoundError(f"No se encontró el preprocesador en {PREPROC_PATH}")
    model = joblib.load(MODEL_PATH)
    pre = joblib.load(PREPROC_PATH)
    return model, pre


def _infer_raw_columns_from_preprocessor(pre) -> Dict[str, List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for name, _trans, cols in pre.transformers_:
        if name == "remainder":
            continue
        if not isinstance(cols, list):
            try:
                if hasattr(pre, "feature_names_in_"):
                    cols = list(pre.feature_names_in_)
                else:
                    cols = []
            except Exception:
                cols = []
        if name == "num":
            numeric_cols.extend([str(c) for c in cols])
        elif name == "cat":
            categorical_cols.extend([str(c) for c in cols])
    return {"numeric": numeric_cols, "categorical": categorical_cols}


try:
    _MODEL, _PREPROC = _load_artifacts()
    _RAW_SCHEMA = _infer_raw_columns_from_preprocessor(_PREPROC)
    _ALIVE = True
except Exception as e:  # noqa: BLE001
    _MODEL, _PREPROC, _RAW_SCHEMA = None, None, {"numeric": [], "categorical": []}
    _ALIVE = False
    _STARTUP_ERROR = str(e)


@app.get("/health")
def health():
    return {
        "ok": _ALIVE,
        "model_path": str(MODEL_PATH),
        "preprocessor_path": str(PREPROC_PATH),
        "error": None if _ALIVE else _STARTUP_ERROR,
    }


@app.get("/schema")
def schema():
    if not _ALIVE:
        raise HTTPException(status_code=503, detail=_STARTUP_ERROR)
    return _RAW_SCHEMA


def _dataframe_from_features(
    features: Dict[str, Any],
    raw_schema: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Construye un DataFrame de UNA fila con las columnas crudas esperadas.
    Llena faltantes con None; deja extras para reportarlas.
    """
    cols = list(raw_schema["numeric"]) + list(raw_schema["categorical"])
    data: Dict[str, List[Any]] = {}
    for c in cols:
        data[c] = [features.get(c, None)]
    df = pd.DataFrame(data)
    return df


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Umbral para clasificar"),
):
    if not _ALIVE:
        raise HTTPException(status_code=503, detail=_STARTUP_ERROR)

    # Construye DF crudo
    df_raw = _dataframe_from_features(req.features, _RAW_SCHEMA)

    # Detecta faltantes y extras (solo informativo)
    expected = set(_RAW_SCHEMA["numeric"] + _RAW_SCHEMA["categorical"])
    got = set(req.features.keys())
    missing = sorted(list(expected - got))
    extra = sorted(list(got - expected))

    # Transforma
    try:
        X = _PREPROC.transform(df_raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Error en preprocesamiento: {e}")

    # Predice
    try:
        proba = float(_MODEL.predict_proba(X)[:, 1][0])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

    label = int(proba >= float(threshold))
    return PredictResponse(
        ok=True,
        proba=proba,
        label=label,
        threshold=float(threshold),
        missing=missing,
        extra=extra,
    )
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model import default_config, make_model


@dataclass
class TrainConfig:
    model_cfg: Dict[str, Any]
    max_epochs: int = 100
    patience: int = 10
    outdir: Path = Path("results")


def _load_xy(csv_path: Path, target: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"No encuentro la columna objetivo '{target}' en {csv_path}")
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y


def evaluate_model(
    model_path: Path,
    test_csv: Path,
    outdir: Path,
    threshold: float = 0.5,
) -> Dict:
    """Evalúa el modelo en test y guarda artefactos."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Carga modelo y datos
    model = joblib.load(model_path)
    Xte, yte = _load_xy(test_csv, target="label")

    # Predicciones
    proba = model.predict_proba(Xte)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # Métricas
    metrics = compute_classification_metrics(yte.values, proba, threshold=threshold)
    conf = confusion_counts(yte.values, y_pred)

    # Gráficos
    plot_roc(yte.values, proba, outdir / "roc_curve.png")
    plot_pr(yte.values, proba, outdir / "pr_curve.png")

    # Guarda JSON
    out = {
        "threshold": float(threshold),
        "metrics": metrics,
        "confusion": conf,
        "n_test": int(len(yte)),
    }
    save_json(out, outdir / "test_metrics.json")
    return out
def _class_weight_sample_weights(y: pd.Series) -> Optional[np.ndarray]:
    """
    Calcula weights por clase para pasar como sample_weight a estimadores
    que no soportan class_weight directamente (como MLPClassifier).
    Devuelve None si hay problema.
    """
    try:
        classes, counts = np.unique(y, return_counts=True)
        freq = dict(zip(classes.tolist(), counts.tolist()))
        if len(freq) < 2:
            return None  # no se puede ponderar una sola clase
        total = len(y)
        # weight_c = total / (num_clases * count_c)
        weights = {c: total / (len(freq) * cnt) for c, cnt in freq.items()}
        return np.array([weights[int(t)] for t in y.tolist()], dtype=float)
    except Exception:
        return None


def _eval_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """
    Calcula AUROC y AUPRC. Maneja casos edge cuando no hay ambas clases.
    """
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


def _save_log(log_rows: list[Dict[str, Any]], out_csv: Path) -> None:
    df = pd.DataFrame(log_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def _save_model(model, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def _is_mlp(model) -> bool:
    return model.__class__.__name__.lower().startswith("mlp")


def train_and_validate(cfg: TrainConfig, data_dir: Path, target: str = "label") -> None:
    # --- carga datos ---
    Xtr, ytr = _load_xy(data_dir / "train.csv", target=target)
    Xva, yva = _load_xy(data_dir / "val.csv", target=target)

    # --- construye modelo ---
    model = make_model(cfg.model_cfg)

    # --- rutas de salida ---
    outdir = cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model.joblib"
    log_path = outdir / "train_log.csv"

    log_rows: list[Dict[str, Any]] = []

    if not _is_mlp(model):
        # ----- LOGREG (u otro estimador no incremental): fit único -----
        model.fit(Xtr, ytr)
        proba_va = model.predict_proba(Xva)[:, 1]
        metrics_va = _eval_metrics(yva.values, proba_va)
        log_rows.append({"epoch": 1, **metrics_va})
        _save_log(log_rows, log_path)
        _save_model(model, model_path)
        print(f"✔ Modelo guardado en: {model_path}")
        print(f"Val AUROC={metrics_va['auroc']:.4f} AUPRC={metrics_va['auprc']:.4f}")
        return

    # ----- MLP con early stopping manual e incremental -----
    # Configuramos para entrenamiento por épocas controladas:
    # - max_iter=1 y warm_start=True para avanzar 1 "época" por fit()
    # - usamos sample_weight para class balance
    params = model.get_params()
    if params.get("max_iter", 1) != 1:
        model.set_params(max_iter=1)
    if not params.get("warm_start", False):
        try:
            model.set_params(warm_start=True)
        except Exception:
            pass  # algunos estimadores podrían no soportar warm_start

    # Ponderación por clase
    sw = _class_weight_sample_weights(ytr)

    best_auroc = -np.inf
    best_epoch = 0
    best_state: Optional[bytes] = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        # Una "época" de actualización
        try:
            model.fit(Xtr, ytr, sample_weight=sw)  # para MLPClassifier es válido
        except TypeError:
            # fallback si no acepta sample_weight
            model.fit(Xtr, ytr)

        # Eval en validación
        proba_va = model.predict_proba(Xva)[:, 1]
        metrics_va = _eval_metrics(yva.values, proba_va)
        log_rows.append({"epoch": epoch, **metrics_va})

        # Early stopping por AUROC
        auroc = metrics_va["auroc"]
        if np.isfinite(auroc) and auroc > best_auroc:
            best_auroc = auroc
            best_epoch = epoch
            # guardamos snapshot binario del modelo (pickle) en memoria
            best_state = joblib.dumps(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d} | Val AUROC={metrics_va['auroc']:.4f} "
            f"AUPRC={metrics_va['auprc']:.4f} | "
            f"Best@{best_epoch}={best_auroc:.4f} | "
            f"no_improve={epochs_no_improve}/{cfg.patience}"
        )

        if epochs_no_improve >= cfg.patience:
            print("⏹ Early stopping activado.")
            break

    # Guardar log y mejor checkpoint
    _save_log(log_rows, log_path)
    if best_state is not None:
        best_model = joblib.loads(best_state)
        _save_model(best_model, model_path)
        print(f"✔ Mejor modelo (epoch {best_epoch}) guardado en: {model_path}")
    else:
        # si nunca mejoró, guardamos el modelo actual
        _save_model(model, model_path)
        print("⚠ No hubo mejora; guardado el modelo final.")

    print(f"📄 Log de entrenamiento: {log_path}")


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
