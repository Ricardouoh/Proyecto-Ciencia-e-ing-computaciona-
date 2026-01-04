from __future__ import annotations
"""
Preprocesamiento tabular mejorado:
- Limpieza previa:
    * Elimina columnas tipo ID (_id, submitter_id, etc.)
    * Elimina columnas con >95% de NaN
    * (Opcional) Elimina categóricas con cardinalidad muy alta
- Split estratificado en train/val/test
- Imputación (num: mediana, cat: "unknown")
- Escalado numérico (z-score)
- One-hot en categóricas (handle_unknown="ignore", min_frequency)
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


# ------------------------- helpers de limpieza ------------------------- #

def clean_binary_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, int]:
    """
    Fuerza el target a binario 0/1 y elimina filas no mapeables.
    Retorna el DF limpio y cuantas filas se eliminaron.
    """
    if target not in df.columns:
        raise ValueError(f"No se encontro la columna target '{target}'")

    def _map(v):
        if pd.isna(v):
            return None
        if isinstance(v, (int, float)) and v in (0, 1):
            return int(v)
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "si", "y", "positivo", "positive", "pos"}:
            return 1
        if s in {"0", "false", "no", "neg", "negative", "n"}:
            return 0
        return None

    df2 = df.copy()
    df2[target] = df2[target].map(_map)
    before = len(df2)
    df2 = df2.dropna(subset=[target])
    df2[target] = df2[target].astype(int)
    dropped = before - len(df2)
    return df2, dropped


def drop_duplicate_rows(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, int]:
    """
    Elimina filas duplicadas para evitar fuga de informacion entre splits.
    Prioriza case_key si existe; si no, deduplica por todas las features.
    """
    if "case_key" in df.columns:
        subset = ["case_key", target]
    else:
        subset = list(df.columns)
    before = len(df)
    df2 = df.drop_duplicates(subset=subset)
    return df2, before - len(df2)


def drop_leakage_columns(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Elimina columnas que contienen informacion directa del target (fuga).
    Ajusta la lista si agregas nuevos derivados del target.
    """
    leak_cols = [
        "dem_vital_status",
        "dem_cause_of_death",
        "dem_days_to_death",
        "any_progression",
    ]
    keep_cols = [c for c in df.columns if c not in leak_cols or c == target]
    dropped = [c for c in df.columns if c not in keep_cols and c != target]
    return df[keep_cols].copy(), dropped


def coerce_boolean_like(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detecta columnas booleanas (texto/objetos) y las convierte a 0/1.
    Solo convierte si >70% de los valores son mapeables.
    """
    bool_map = {
        "1": 1, "true": 1, "yes": 1, "si": 1, "y": 1, "positivo": 1, "positive": 1, "pos": 1,
        "0": 0, "false": 0, "no": 0, "n": 0, "neg": 0, "negative": 0,
    }
    converted: List[str] = []
    df2 = df.copy()

    for c in df.columns:
        if c == target or pd.api.types.is_numeric_dtype(df2[c]):
            continue
        if pd.api.types.is_bool_dtype(df2[c]):
            df2[c] = df2[c].astype(int)
            converted.append(c)
            continue

        s = df2[c].astype(str).str.lower().str.strip()
        mapped = s.map(bool_map)
        mappable_ratio = mapped.notna().mean()
        if mappable_ratio >= 0.7:
            df2[c] = mapped.astype("Int64").astype(float)
            converted.append(c)

    return df2, converted


def convert_datetime_columns(
    df: pd.DataFrame,
    target: str,
    explicit_date_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convierte columnas de fecha a un valor numerico (dias desde epoch) para el MLP.
    Detecta: columnas datetime64 o que contengan 'date'/'time' en el nombre.
    """
    df2 = df.copy()
    converted: List[str] = []

    candidates: List[str] = []
    if explicit_date_cols:
        candidates.extend([c for c in explicit_date_cols if c in df2.columns and c != target])
    for c in df2.columns:
        if c == target:
            continue
        name = c.lower()
        if pd.api.types.is_datetime64_any_dtype(df2[c]) or ("date" in name) or ("time" in name):
            candidates.append(c)

    for c in candidates:
        try:
            dt = pd.to_datetime(
                df2[c],
                errors="coerce",
                utc=True,
                format="mixed",
            )
        except Exception:
            continue
        if dt.notna().mean() < 0.3:
            continue
        df2[c] = dt.astype("int64") / 86_400_000_000_000  # dias
        converted.append(c)

    return df2, converted


def add_age_from_days(df: pd.DataFrame, source_col: str = "mean_age_at_dx") -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Si existe source_col (edad en dias), crea una columna en años y elimina la original.
    """
    if source_col not in df.columns:
        return df, None
    df2 = df.copy()
    df2[source_col] = pd.to_numeric(df2[source_col], errors="coerce")
    new_col = f"{source_col}_years"
    df2[new_col] = df2[source_col] / 365.25
    df2 = df2.drop(columns=[source_col])
    return df2, new_col


def add_ajcc_stage_ordinal(df: pd.DataFrame, col: str = "last_ajcc_stage") -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Mapea la etapa AJCC a un valor ordinal creciente (0 < I < II < III < IV).
    """
    if col not in df.columns:
        return df, None
    order = [
        "stage 0", "stage i", "stage ia", "stage ib", "stage ic",
        "stage ii", "stage iia", "stage iib", "stage iic",
        "stage iii", "stage iiia", "stage iiib", "stage iiic",
        "stage iv", "stage iva", "stage ivb", "stage ivc",
    ]
    mapping = {name: idx for idx, name in enumerate(order)}
    df2 = df.copy()
    stage_norm = df2[col].astype(str).str.lower().str.strip()
    df2[f"{col}_ord"] = stage_norm.map(mapping)
    df2 = df2.drop(columns=[col])
    return df2, f"{col}_ord"


def drop_constant_columns(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Elimina columnas con un solo valor (incluyendo NaN) para reducir ruido.
    """
    to_drop: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].nunique(dropna=False) <= 1:
            to_drop.append(c)
    df2 = df.drop(columns=to_drop, errors="ignore")
    return df2, to_drop


def drop_id_like_columns(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Elimina columnas tipo ID que generarán cardinalidad absurda:
    - columnas que terminan en '_id'
    - columnas explícitas como 'submitter_id', 'case_submitter_id'
    """
    to_drop: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        if c.endswith("_id") or c in ("submitter_id", "case_submitter_id"):
            to_drop.append(c)
    df2 = df.drop(columns=to_drop, errors="ignore")
    return df2, to_drop


def drop_high_nan_columns(df: pd.DataFrame, target: str, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Elimina columnas donde la proporción de NaN sea > threshold (por defecto 95%).
    No toca la columna target.
    """
    to_drop: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].isna().mean() > threshold:
            to_drop.append(c)
    df2 = df.drop(columns=to_drop, errors="ignore")
    return df2, to_drop


def drop_high_cardinality_categoricals(
    df: pd.DataFrame,
    target: str,
    max_unique: int = 500,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Elimina columnas categóricas con demasiadas categorías distintas
    (por defecto >500). Esto evita explosión de one-hot en columnas locas
    que no sean IDs pero sí tengan cardinalidad extrema.

    Si quieres desactivar esto, sube max_unique a un número ridículamente alto.
    """
    to_drop: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            try:
                nunique = df[c].nunique(dropna=True)
            except TypeError:
                continue
            if nunique > max_unique:
                to_drop.append(c)
    df2 = df.drop(columns=to_drop, errors="ignore")
    return df2, to_drop


# ------------------------- split & columnas ------------------------- #

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


# ------------------------- pipelines ------------------------- #

def make_numeric_pipeline():
    """Imputación mediana + escalado estándar."""
    from sklearn.pipeline import Pipeline
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def make_categorical_pipeline():
    """Imputación 'unknown' + one-hot (ignora categorías nuevas, agrupa raras)."""
    from sklearn.pipeline import Pipeline

    # Intentar usar min_frequency si la versión de sklearn lo soporta
    # (agrupa categorías raras y reduce columnas).
    ohe = None
    for kwargs in (
        dict(handle_unknown="ignore", sparse_output=False, min_frequency=0.01),
        dict(handle_unknown="ignore", sparse=False, min_frequency=0.01),
        dict(handle_unknown="ignore", sparse_output=False),
        dict(handle_unknown="ignore", sparse=False),
    ):
        try:
            ohe = OneHotEncoder(**kwargs)
            break
        except TypeError:
            continue
    if ohe is None:
        # último fallback (muy improbable)
        ohe = OneHotEncoder(handle_unknown="ignore")

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


# ------------------------- ajuste & transformación ------------------------- #

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


# ------------------------- CLI ------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Preprocesa CSV tabular (limpieza + split + imputación + escala + OHE) y guarda artefactos."
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
    ap.add_argument(
        "--nan-threshold",
        type=float,
        default=0.95,
        help="Umbral para eliminar columnas con proporción de NaN > threshold (default=0.95).",
    )
    ap.add_argument(
        "--max-unique-cat",
        type=int,
        default=500,
        help="Máximo de categorías únicas permitido antes de eliminar una columna categórica (default=500).",
    )

    args = ap.parse_args()

    # Leer CSV (parseo opcional de fechas)
    date_cols = [c.strip() for c in args.parse_dates.split(",") if c.strip()]
    read_kwargs = {}
    if date_cols:
        read_kwargs["parse_dates"] = date_cols

    df_in = pd.read_csv(args.csv, **read_kwargs)
    target_col = args.target

    # ---------------- Sanitizacion del target y duplicados ---------------- #

    df_in, dropped_target = clean_binary_target(df_in, target_col)
    if dropped_target:
        print("!! Filas eliminadas por target invalido:", dropped_target)

    df_in, dropped_dups = drop_duplicate_rows(df_in, target_col)
    if dropped_dups:
        print("!! Filas duplicadas eliminadas:", dropped_dups)

    # Booleanos a 0/1 para vectorizacion densa
    df_in, bool_cols = coerce_boolean_like(df_in, target_col)
    if bool_cols:
        print("✔ Columnas convertidas a booleano (0/1):", ", ".join(bool_cols))

    # Fechas a numerico (dias)
    df_in, conv_dates = convert_datetime_columns(df_in, target_col, explicit_date_cols=date_cols)
    if conv_dates:
        print("✔ Columnas de fecha convertidas a dias:", ", ".join(conv_dates))

    df_in, dropped_leaks = drop_leakage_columns(df_in, target_col)
    if dropped_leaks:
        print("!! Columnas eliminadas por fuga de target:", ", ".join(dropped_leaks))

    # Features derivadas: edad en años y etapa AJCC ordinal
    df_in, age_col = add_age_from_days(df_in, source_col="mean_age_at_dx")
    if age_col:
        print("✔ Columna derivada de edad (años):", age_col)

    df_in, stage_col = add_ajcc_stage_ordinal(df_in, col="last_ajcc_stage")
    if stage_col:
        print("✔ Columna ordinal creada para etapa AJCC:", stage_col)

    # ---------------- Limpieza previa antes de inferir columnas ---------------- #

    dropped_ids: List[str] = []
    dropped_nan: List[str] = []
    dropped_card: List[str] = []
    dropped_const: List[str] = []

    # 1) Drop columnas ID-like
    df_in, dropped = drop_id_like_columns(df_in, target_col)
    dropped_ids.extend(dropped)

    # 2) Drop columnas con demasiados NaN
    df_in, dropped = drop_high_nan_columns(df_in, target_col, threshold=float(args.nan_threshold))
    dropped_nan.extend(dropped)

    # 3) Drop categóricas con cardinalidad muy alta
    df_in, dropped = drop_high_cardinality_categoricals(
        df_in,
        target_col,
        max_unique=int(args.max_unique_cat),
    )
    dropped_card.extend(dropped)

    # 4) Drop columnas constantes
    df_in, dropped = drop_constant_columns(df_in, target_col)
    dropped_const.extend(dropped)

    if dropped_ids:
        print("ℹ️ Columnas tipo ID eliminadas:", ", ".join(dropped_ids))
    if dropped_nan:
        print("ℹ️ Columnas con alta proporción de NaN eliminadas:", ", ".join(dropped_nan))
    if dropped_card:
        print("ℹ️ Columnas categóricas de cardinalidad muy alta eliminadas:", ", ".join(dropped_card))
    if dropped_const:
        print("ℹ️ Columnas constantes eliminadas:", ", ".join(dropped_const))

    # ---------------- Inferencia (o hints) de columnas ---------------- #

    num_hint = [c.strip() for c in args.numeric.split(",") if c.strip()]
    cat_hint = [c.strip() for c in args.categorical.split(",") if c.strip()]
    numeric_cols, categorical_cols = infer_columns(
        df_in, target=target_col,
        numeric_hint=num_hint or None,
        categorical_hint=cat_hint or None,
    )

    # ---------------- Split ---------------- #

    split = stratified_split(
        df_in, target=target_col,
        val_size=float(args.val_size),
        test_size=float(args.test_size),
        random_state=int(args.seed),
    )

    # ---------------- Transform ---------------- #

    outdir = Path(args.outdir)
    Xtr, Xva, Xte = fit_transform_all(
        split, numeric_cols, categorical_cols,
        preprocessor_path=outdir / "preprocessor.joblib",
    )

    outdir.mkdir(parents=True, exist_ok=True)

    # Guardar CSVs con target al final
    pd.concat([Xtr.reset_index(drop=True), split.y_train.reset_index(drop=True)], axis=1).to_csv(
        outdir / "train.csv", index=False
    )
    pd.concat([Xva.reset_index(drop=True), split.y_val.reset_index(drop=True)], axis=1).to_csv(
        outdir / "val.csv", index=False
    )
    pd.concat([Xte.reset_index(drop=True), split.y_test.reset_index(drop=True)], axis=1).to_csv(
        outdir / "test.csv", index=False
    )

    print("✔ Guardado preprocesador en:", outdir / "preprocessor.joblib")
    print("✔ train/val/test en:", outdir)
    print(f"   - train.csv: {len(split.y_train)} filas, {len(Xtr.columns)} cols")
    print(f"   - val.csv:   {len(split.y_val)} filas, {len(Xva.columns)} cols")
    print(f"   - test.csv:  {len(split.y_test)} filas, {len(Xte.columns)} cols")
