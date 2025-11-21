from __future__ import annotations
"""
Preprocesamiento tabular:
- Split estratificado en train/val/test
- Imputación (num: mediana, cat: "unknown")
- Escalado numérico (z-score)
- One-hot en categóricas (handle_unknown="ignore")
- Guarda preprocesador (joblib) y CSVs procesados
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

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
    if target not in df.columns:
        raise ValueError(f"No encuentro la columna objetivo '{target}' en el CSV de entrada.")
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=val_size + test_size,
        stratify=y,
        random_state=random_state,
    )
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=rel_test,
        stratify=y_tmp,
        random_state=random_state,
    )
    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test)


def infer_columns(
    df: pd.DataFrame,
    target: str,
    numeric_hint: Optional[List[str]] = None,
    categorical_hint: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Infiera columnas numéricas y categóricas (excluye el target).
    Puedes forzar listas con *_hint* (se filtran a columnas existentes).
    """
    cols = [c for c in df.columns if c != target]

    if numeric_hint is not None or categorical_hint is not None:
        num = [c for c in (numeric_hint or []) if c in cols]
        cat = [c for c in (categorical_hint or []) if c in cols]
        # lo que no quedó en num ni cat, asúmelo categórico
        rest = [c for c in cols if c not in num and c not in cat]
        cat.extend(rest)
        return num, cat

    # heurística por dtype
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    return num, cat


def make_numeric_pipeline():
    """Imputación mediana + escalado estándar."""
    from sklearn.pipeline import Pipeline
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def make_categorical_pipeline():
    """Imputación 'unknown' + one-hot (ignora categorías nuevas).

    Compatible con distintas versiones de scikit-learn:
    - >=1.2 usa `sparse_output=False`
    - < 1.2 usa `sparse=False`
    """
    from sklearn.pipeline import Pipeline
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", ohe),
    ])


def make_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """ColumnTransformer con pipelines numéricos y categóricos."""
    num_pipe = make_numeric_pipeline()
    cat_pipe = make_categorical_pipeline()
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre


def fit_transform_all(
    split: SplitData,
    numeric_cols: List[str],
    categorical_cols: List[str],
    preprocessor_path: str | Path = "results/preprocessor.joblib",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ajusta el preprocesador en train y transforma train/val/test."""
    pre = make_preprocessor(numeric_cols, categorical_cols)

    Xtr = pre.fit_transform(split.X_train)
    Xva = pre.transform(split.X_val)
    Xte = pre.transform(split.X_test)

    # nombres de columnas resultantes (num + one-hot)
    ohe = pre.named_transformers_["cat"]["ohe"]  # type: ignore[index]
    cat_names = list(ohe.get_feature_names_out(categorical_cols))
    num_names = list(numeric_cols)
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

    ap = argparse.ArgumentParser(
        description="Preprocesa CSV tabular (split + imputación + escala + OHE) y guarda artefactos."
    )
    ap.add_argument("--csv", required=True, help="Ruta al CSV de entrada (features + target).")
    ap.add_argument("--target", default="label", help="Nombre de la columna objetivo (binaria).")
    ap.add_argument("--outdir", default="data/processed", help="Carpeta de salida para CSVs y preprocesador.")
    ap.add_argument("--val-size", type=float, default=0.15, help="Proporción de validación (default=0.15).")
    ap.add_argument("--test-size", type=float, default=0.15, help="Proporción de test (default=0.15).")
    ap.add_argument("--seed", type=int, default=42, help="Random state.")
    ap.add_argument("--numeric", default="", help="Lista de columnas numéricas separadas por coma (opcional).")
    ap.add_argument("--categorical", default="", help="Lista de columnas categóricas separadas por coma (opcional).")
    ap.add_argument("--parse-dates", default="", help="Columnas de fecha a parsear (coma, opcional).")

    args = ap.parse_args()

    # Leer CSV (parseo opcional de fechas)
    date_cols = [c.strip() for c in args.parse_dates.split(",") if c.strip()]
    read_kwargs = {}
    if date_cols:
        read_kwargs["parse_dates"] = date_cols

    df_in = pd.read_csv(args.csv, **read_kwargs)

    # Inferencia (o hints) de columnas
    num_hint = [c.strip() for c in args.numeric.split(",") if c.strip()]
    cat_hint = [c.strip() for c in args.categorical.split(",") if c.strip()]
    numeric_cols, categorical_cols = infer_columns(
        df_in, target=args.target,
        numeric_hint=num_hint or None,
        categorical_hint=cat_hint or None,
    )

    # Split
    split = stratified_split(
        df_in, target=args.target,
        val_size=float(args.val_size),
        test_size=float(args.test_size),
        random_state=int(args.seed),
    )

    # Transform
    Xtr, Xva, Xte = fit_transform_all(
        split, numeric_cols, categorical_cols,
        preprocessor_path=Path(args.outdir) / "preprocessor.joblib",
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Guardar CSVs con target al final
    pd.concat([Xtr, split.y_train.reset_index(drop=True)], axis=1).to_csv(outdir / "train.csv", index=False)
    pd.concat([Xva, split.y_val.reset_index(drop=True)], axis=1).to_csv(outdir / "val.csv", index=False)
    pd.concat([Xte, split.y_test.reset_index(drop=True)], axis=1).to_csv(outdir / "test.csv", index=False)

    print("✔ Guardado preprocesador en:", outdir / "preprocessor.joblib")
    print("✔ train/val/test en:", outdir)
    print(f"   - train.csv: {len(split.y_train)} filas, {len(Xtr.columns)} cols")
    print(f"   - val.csv:   {len(split.y_val)} filas, {len(Xva.columns)} cols")
    print(f"   - test.csv:  {len(split.y_test)} filas, {len(Xte.columns)} cols")
