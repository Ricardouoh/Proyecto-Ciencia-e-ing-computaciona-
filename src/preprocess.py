from __future__ import annotations
"""
Preprocesamiento tabular:
- Split estratificado en train/val/test
- Imputación (num: mediana, cat: "unknown")
- Escalado numérico (z-score)
- One-hot en categóricas (handle_unknown="ignore")
- Guardado/carga del preprocesador con joblib
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def stratified_split(
    df: pd.DataFrame,
    target: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> SplitData:
    """Hace split estratificado en train/val/test manteniendo proporciones."""
    y = df[target]
    X = df.drop(columns=[target])

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=val_size + test_size,
        stratify=y,
        random_state=random_state,
    )
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=rel_test,
        stratify=y_tmp,
        random_state=random_state,
    )
    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test)


def infer_columns(
    df: pd.DataFrame,
    target: str,
    numeric_hint: List[str] | None = None,
    categorical_hint: List[str] | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Infiera columnas numéricas y categóricas.
    Puedes forzar con *_hint*. El target se excluye siempre.
    """
    cols = [c for c in df.columns if c != target]
    if numeric_hint is not None or categorical_hint is not None:
        num = [c for c in (numeric_hint or []) if c in cols]
        cat = [c for c in (categorical_hint or []) if c in cols]
        rest = [c for c in cols if c not in num and c not in cat]
        # resto: asume categórico
        cat.extend(rest)
        return num, cat

    # inferencia simple por dtype
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    return num, cat


def make_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Crea el ColumnTransformer con imputación + escalado/one-hot."""
    num_pipe = make_numeric_pipeline()
    cat_pipe = make_categorical_pipeline()
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )
    return pre


def make_numeric_pipeline():
    """Imputación mediana + escalado estándar."""
    from sklearn.pipeline import Pipeline
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def make_categorical_pipeline():
    """Imputación 'unknown' + one-hot (ignora categorías nuevas)."""
    from sklearn.pipeline import Pipeline
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def fit_transform_all(
    split: SplitData,
    numeric_cols: List[str],
    categorical_cols: List[str],
    preprocessor_path: str | Path = "results/preprocessor.joblib",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ajusta el preprocesador en train y transforma train/val/test.
    Guarda el preprocesador en disco para inferencia posterior.
    """
    pre = make_preprocessor(numeric_cols, categorical_cols)

    Xtr = pre.fit_transform(split.X_train)
    Xva = pre.transform(split.X_val)
    Xte = pre.transform(split.X_test)

    # nombres de columnas resultantes
    ohe = pre.named_transformers_["cat"]["ohe"]  # type: ignore[index]
    cat_names = list(ohe.get_feature_names_out(categorical_cols))
    num_names = numeric_cols
    out_cols = num_names + cat_names

    Xtr = pd.DataFrame(Xtr, columns=out_cols, index=split.X_train.index)
    Xva = pd.DataFrame(Xva, columns=out_cols, index=split.X_val.index)
    Xte = pd.DataFrame(Xte, columns=out_cols, index=split.X_test.index)

    Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, preprocessor_path)

    return Xtr, Xva, Xte


def load_preprocessor(preprocessor_path: str | Path):
    """Carga el preprocesador guardado en disco."""
    return joblib.load(preprocessor_path)


def transform_with_loaded(
    preprocessor,
    X: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    """Transforma un DataFrame con un preprocesador ya cargado."""
    Xt = preprocessor.transform(X)
    ohe = preprocessor.named_transformers_["cat"]["ohe"]  # type: ignore[index]
    cat_names = list(ohe.get_feature_names_out(categorical_cols))
    out_cols = list(numeric_cols) + cat_names
    return pd.DataFrame(Xt, columns=out_cols, index=X.index)



if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Evaluación en test.")
    ap.add_argument(
        "--model",
        default="results/model.joblib",
        help="Ruta al modelo entrenado (joblib).",
    )
    ap.add_argument(
        "--test-csv",
        default="data/processed/test.csv",
        help="CSV de test (features + label).",

    ap = argparse.ArgumentParser(
        description="Entrenamiento con validación y early stopping (MLP)."
    )
    ap.add_argument(
        "--data-dir",
        default="data/processed",
        help="Carpeta con train.csv y val.csv (por defecto data/processed).",
    )
    ap.add_argument(
        "--config",
        default="",
        help="Ruta a JSON con configuración del modelo (opcional).",
    )
    ap.add_argument(
        "--model",
        default="mlp",
        choices=["mlp", "logreg"],
        help="Modelo por defecto si no se provee config JSON.",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Máximo de épocas para MLP (default=100).",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Paciencia para early stopping por AUROC (default=10).",
    )
    ap.add_argument(
        "--outdir",
        default="results",
        help="Carpeta de salida (gráficos y JSON).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral para métricas a clase (default=0.5).",
    )
    args = ap.parse_args()

    summary = evaluate_model(
        model_path=Path(args.model),
        test_csv=Path(args.test_csv),
        outdir=Path(args.outdir),
        threshold=float(args.threshold),
    )
    print(json.dumps(summary, indent=2))
        help="Carpeta de salida para modelo y logs (default=results).",
    )
    args = ap.parse_args()

    # carga cfg
    cfg_json = _load_json(args.config)
    if cfg_json:
        model_cfg = cfg_json
    else:
        model_cfg = default_config(args.model)

    train_cfg = TrainConfig(
        model_cfg=model_cfg,
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        outdir=Path(args.outdir),
    )

    train_and_validate(train_cfg, Path(args.data_dir))
        description="Preprocesa CSV tabular (split + transform) y guarda artefactos.",
    )
    ap.add_argument("--csv", required=True, help="Ruta al CSV de entrada.")
    ap.add_argument("--target", default="label", help="Nombre de la columna objetivo.")
    ap.add_argument(
        "--outdir",
        default="data/processed",
        help="Carpeta de salida para CSVs y preprocesador.",
    )
    args = ap.parse_args()

    df_in = pd.read_csv(args.csv)
    num_cols, cat_cols = infer_columns(df_in, target=args.target)

    split = stratified_split(df_in, target=args.target)
    Xtr, Xva, Xte = fit_transform_all(
        split, num_cols, cat_cols, preprocessor_path=Path(args.outdir) / "preprocessor.joblib"
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.concat([Xtr, split.y_train], axis=1).to_csv(outdir / "train.csv", index=False)
    pd.concat([Xva, split.y_val], axis=1).to_csv(outdir / "val.csv", index=False)
    pd.concat([Xte, split.y_test], axis=1).to_csv(outdir / "test.csv", index=False)

    print("✔ Guardado preprocesador en:", outdir / "preprocessor.joblib")
    print("✔ train/val/test en:", outdir)
