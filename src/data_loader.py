# src/data_loader.py
from __future__ import annotations
"""
Data Loader generico para tabulares.

Soporta: CSV, JSONL/NDJSON, Parquet, Excel (XLS/XLSX), SAS XPT.
- Carga 1 archivo o una carpeta completa (concatena por filas o merge por clave).
- Normaliza nombres de columnas (snake_case).
- Opcional: elimina columnas constantes/duplicadas, filas duplicadas.
- Intenta castear numéricos y fechas.
- Valida la columna target si se provee.
- Guarda a CSV final (UTF-8) o devuelve DataFrame.

Uso CLI (ejemplos):
  # Un archivo CSV -> CSV unificado
  python -m src.data_loader --in data/raw/pacientes.csv --out data/training/raw.csv --target label

  # Carpeta con varios archivos mezclados
  python -m src.data_loader --in data/raw_folder --out data/training/raw.csv --target label --drop-const --drop-dup-rows
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


# ------------------------- utilidades ------------------------- #

def _to_snake(name: str) -> str:
    out = (
        name.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("__", "_")
        .lower()
    )
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_to_snake(str(c)) for c in df.columns]
    return df


def _try_numeric_cast(df: pd.DataFrame, ignore: Iterable[str] = ()) -> pd.DataFrame:
    """Intenta convertir columnas a numérico donde aplique."""
    df = df.copy()
    for c in df.columns:
        if c in ignore:
            continue
        # evita castear ids alfanuméricos largos
        if df[c].dtype == object:
            ser = pd.to_numeric(df[c], errors="ignore")
            # si la conversión reduce objetos, úsala
            if getattr(ser, "dtype", None) is not None and ser.dtype != object:
                df[c] = ser
    return df


def _try_datetime_cast(df: pd.DataFrame, date_cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    return df


def _drop_constant_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dropped: List[str] = []
    keep = []
    for c in df.columns:
        # nunique(dropna=False) cuenta NaN como categoría
        if df[c].nunique(dropna=False) <= 1:
            dropped.append(c)
        else:
            keep.append(c)
    return df[keep].copy(), dropped


def _dedupe_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Elimina columnas duplicadas por contenido exacto."""
    dropped: List[str] = []
    seen = {}
    keep_cols: List[str] = []
    for c in df.columns:
        key = (tuple(pd.Series(df[c]).fillna("__NaN__")), str(df[c].dtype))
        if key in seen:
            dropped.append(c)
        else:
            seen[key] = c
            keep_cols.append(c)
    return df[keep_cols].copy(), dropped


def _filter_never_cancer(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Filtra registros con MCQ220 == 2 (nunca les dijeron que tienen cancer).
    Si no existe la columna, no hace nada.
    """
    if "mcq220" not in df.columns:
        return df, 0
    ser = df["mcq220"]
    ser_num = pd.to_numeric(ser, errors="coerce")
    mask_num = ser_num == 2
    ser_txt = ser.astype(str).str.lower().str.strip()
    mask_txt = ser_txt.isin({"2", "no", "never", "no (2)"})
    mask = mask_num | mask_txt
    before = len(df)
    df2 = df.loc[mask].copy()
    return df2, before - len(df2)


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".csv",):
        # intenta codificaciones comunes
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        # último intento por defecto
        return pd.read_csv(path)
    if suf in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if suf in (".parquet",):
        return pd.read_parquet(path)
    if suf in (".xls", ".xlsx"):
        return pd.read_excel(path)
    if suf in (".xpt", ".xps"):
        return pd.read_sas(path, format="xport")
    raise ValueError(f"Formato no soportado: {path.name}")


def _list_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files: List[Path] = []
    for suf in ("*.csv", "*.jsonl", "*.ndjson", "*.parquet", "*.xls", "*.xlsx", "*.xpt", "*.xps"):
        files.extend(sorted(input_path.rglob(suf)))
    if not files:
        raise FileNotFoundError(f"No encontré archivos soportados en {input_path}")
    return files


# ------------------------- API principal ------------------------- #

def load_tabular(
    source: str | Path,
    *,
    target: Optional[str] = None,
    date_cols: Optional[List[str]] = None,
    merge_on: Optional[List[str]] = None,
    merge_how: str = "outer",
    drop_const: bool = False,
    drop_dup_cols: bool = True,
    drop_dup_rows: bool = False,
    keep_columns: Optional[List[str]] = None,
    filter_never_cancer: bool = False,
) -> pd.DataFrame:
    """
    Carga un archivo o carpeta, concatena por filas o hace merge por clave, y normaliza.
    - target: valida presencia (no altera su contenido).
    - date_cols: castea a datetime (si existen).
    - merge_on: si se entrega, hace merge por esas columnas (snake_case).
    - merge_how: tipo de merge (outer/inner/left/right).
    - drop_const: elimina columnas constantes.
    - drop_dup_cols: elimina columnas duplicadas exactas.
    - drop_dup_rows: elimina filas duplicadas exactas (post-normalización).
    - keep_columns: si se entrega, recorta a ese subconjunto si existen.
    - filter_never_cancer: mantiene solo MCQ220==2 (nunca les dijeron que tienen cancer).
    """
    src = Path(source)
    files = _list_files(src)

    frames: List[pd.DataFrame] = []
    for p in files:
        df = _read_any(p)
        df = _normalize_columns(df)
        frames.append(df)

    merge_cols = [_to_snake(c) for c in merge_on] if merge_on else []

    if merge_cols and len(frames) > 1:
        for idx, df in enumerate(frames):
            missing = [c for c in merge_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Faltan columnas para merge {missing} en: {files[idx].name}")
        df_all = frames[0]
        for i, df in enumerate(frames[1:], start=1):
            df_all = df_all.merge(
                df,
                how=merge_how,
                on=merge_cols,
                suffixes=("", f"__{i}"),
            )
    else:
        # Alinea por columnas (outer join), luego rellena solo estructura
        df_all = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # Casteos suaves
    df_all = _try_numeric_cast(df_all, ignore=(target,) if target else ())
    if date_cols:
        df_all = _try_datetime_cast(df_all, date_cols)

    if filter_never_cancer:
        df_all, dropped_filter = _filter_never_cancer(df_all)
        if dropped_filter:
            print(f"?? Filas eliminadas por filtro MCQ220!=2: {dropped_filter}")

    # Recortes y limpieza opcional
    if keep_columns:
        keep = [c for c in keep_columns if c in df_all.columns]
        df_all = df_all[keep].copy()

    dropped_cols_info: List[str] = []
    if drop_dup_cols:
        df_all, dup_cols = _dedupe_columns(df_all)
        if dup_cols:
            dropped_cols_info.extend([f"duplicada:{c}" for c in dup_cols])

    if drop_const:
        df_all, const_cols = _drop_constant_cols(df_all)
        if const_cols:
            dropped_cols_info.extend([f"constante:{c}" for c in const_cols])

    if drop_dup_rows:
        before = len(df_all)
        df_all = df_all.drop_duplicates(ignore_index=True)
        _ = before - len(df_all)

    # Validación de target si se pidió
    if target and target not in df_all.columns:
        raise ValueError(f"No encontré la columna objetivo '{target}' en los datos cargados.")

    # Orden amigable: target al final
    if target and target in df_all.columns:
        cols = [c for c in df_all.columns if c != target] + [target]
        df_all = df_all[cols]

    # Info mínima
    if dropped_cols_info:
        print("INFO Columnas eliminadas:", ", ".join(dropped_cols_info))

    return df_all


def save_csv(df: pd.DataFrame, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


# ------------------------- CLI ------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Carga uno o varios archivos tabulares y genera un CSV unificado."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Ruta a archivo o carpeta.")
    ap.add_argument("--out", dest="out", required=True, help="CSV de salida (p.ej. data/training/raw.csv).")
    ap.add_argument("--target", default=None, help="Nombre de la columna objetivo (opcional).")
    ap.add_argument(
        "--date-cols",
        default="",
        help="Columnas fecha separadas por coma (si existen se castean a datetime).",
    )
    ap.add_argument(
        "--merge-on",
        default="",
        help="Columnas clave para merge (coma). Si se usa, mergea archivos en vez de concatenar filas.",
    )
    ap.add_argument(
        "--merge-how",
        default="outer",
        choices=["outer", "inner", "left", "right"],
        help="Tipo de merge cuando se usa --merge-on.",
    )
    ap.add_argument(
        "--drop-const",
        action="store_true",
        help="Eliminar columnas constantes.",
    )
    ap.add_argument(
        "--no-drop-dup-cols",
        action="store_true",
        help="No eliminar columnas duplicadas exactas (por defecto se eliminan).",
    )
    ap.add_argument(
        "--drop-dup-rows",
        action="store_true",
        help="Eliminar filas duplicadas exactas.",
    )
    ap.add_argument(
        "--keep",
        default="",
        help="Subconjunto de columnas a conservar (separadas por coma).",
    )
    ap.add_argument(
        "--filter-never-cancer",
        action="store_true",
        help="Mantener solo registros con MCQ220==2 (nunca les dijeron que tienen cancer).",
    )

    args = ap.parse_args()

    date_cols: List[str] = [c.strip() for c in args.date_cols.split(",") if c.strip()]
    merge_on: List[str] = [c.strip() for c in args.merge_on.split(",") if c.strip()]
    keep_cols: List[str] = [c.strip() for c in args.keep.split(",") if c.strip()]

    df_out = load_tabular(
        args.inp,
        target=args.target or None,
        date_cols=date_cols or None,
        merge_on=merge_on or None,
        merge_how=str(args.merge_how),
        drop_const=bool(args.drop_const),
        drop_dup_cols=not bool(args.no_drop_dup_cols),
        drop_dup_rows=bool(args.drop_dup_rows),
        keep_columns=keep_cols or None,
        filter_never_cancer=bool(args.filter_never_cancer),
    )
    save_csv(df_out, args.out)
    print(f"OK CSV unificado guardado en: {args.out}  (filas={len(df_out)}, cols={len(df_out.columns)})")
