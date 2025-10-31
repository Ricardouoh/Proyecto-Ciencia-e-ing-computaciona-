# src/data_loader.py
from __future__ import annotations
"""
Data Loader genérico para tabulares.

Soporta: CSV, JSONL/NDJSON, Parquet, Excel (XLS/XLSX).
- Carga 1 archivo o una carpeta completa (concatena por columnas).
- Normaliza nombres de columnas (snake_case).
- Opcional: elimina columnas constantes/duplicadas, filas duplicadas.
- Intenta castear numéricos y fechas.
- Valida la columna target si se provee.
- Guarda a CSV final (UTF-8) o devuelve DataFrame.

Uso CLI (ejemplos):
  # Un archivo CSV -> CSV unificado
  python -m src.data_loader --in data/raw/pacientes.csv --out data/raw.csv --target label

  # Carpeta con varios archivos mezclados
  python -m src.data_loader --in data/raw_folder --out data/raw.csv --target label --drop-const --drop-dup-rows
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
    raise ValueError(f"Formato no soportado: {path.name}")


def _list_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files: List[Path] = []
    for suf in ("*.csv", "*.jsonl", "*.ndjson", "*.parquet", "*.xls", "*.xlsx"):
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
    drop_const: bool = False,
    drop_dup_cols: bool = True,
    drop_dup_rows: bool = False,
    keep_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Carga un archivo o carpeta, concatena por columnas y normaliza.
    - target: valida presencia (no altera su contenido).
    - date_cols: castea a datetime (si existen).
    - drop_const: elimina columnas constantes.
    - drop_dup_cols: elimina columnas duplicadas exactas.
    - drop_dup_rows: elimina filas duplicadas exactas (post-normalización).
    - keep_columns: si se entrega, recorta a ese subconjunto si existen.
    """
    src = Path(source)
    files = _list_files(src)

    frames: List[pd.DataFrame] = []
    for p in files:
        df = _read_any(p)
        df = _normalize_columns(df)
        frames.append(df)

    # Alinea por columnas (outer join), luego rellena sólo estructura
    df_all = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # Casteos suaves
    df_all = _try_numeric_cast(df_all, ignore=(target,) if target else ())
    if date_cols:
        df_all = _try_datetime_cast(df_all, date_cols)

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
        print("ℹ️ Columnas eliminadas:", ", ".join(dropped_cols_info))

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
    ap.add_argument("--out", dest="out", required=True, help="CSV de salida (p.ej. data/raw.csv).")
    ap.add_argument("--target", default=None, help="Nombre de la columna objetivo (opcional).")
    ap.add_argument(
        "--date-cols",
        default="",
        help="Columnas fecha separadas por coma (si existen se castean a datetime).",
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

    args = ap.parse_args()

    date_cols: List[str] = [c.strip() for c in args.date_cols.split(",") if c.strip()]
    keep_cols: List[str] = [c.strip() for c in args.keep.split(",") if c.strip()]

    df_out = load_tabular(
        args.inp,
        target=args.target or None,
        date_cols=date_cols or None,
        drop_const=bool(args.drop_const),
        drop_dup_cols=not bool(args.no_drop_dup_cols),
        drop_dup_rows=bool(args.drop_dup_rows),
        keep_columns=keep_cols or None,
    )
    save_csv(df_out, args.out)
    print(f"✔ CSV unificado guardado en: {args.out}  (filas={len(df_out)}, cols={len(df_out.columns)})")
