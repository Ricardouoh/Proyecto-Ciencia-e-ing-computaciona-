from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def drop_high_nan_columns(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, List[str]]:
    to_drop = [c for c in df.columns if df[c].isna().mean() > threshold]
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    to_drop = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def drop_id_columns(df: pd.DataFrame, id_cols: List[str]) -> tuple[pd.DataFrame, List[str]]:
    present = [c for c in id_cols if c in df.columns]
    return df.drop(columns=present, errors="ignore"), present


def drop_high_nan_rows(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    mask = df.isna().mean(axis=1) <= threshold
    dropped = int((~mask).sum())
    return df.loc[mask].copy(), dropped


def drop_rows_by_nan_count(df: pd.DataFrame, max_nan: int) -> tuple[pd.DataFrame, int]:
    counts = df.isna().sum(axis=1)
    mask = counts <= max_nan
    dropped = int((~mask).sum())
    return df.loc[mask].copy(), dropped


def clean_nhanes(
    inp: str | Path,
    out: str | Path,
    *,
    col_nan_threshold: float = 0.95,
    row_nan_threshold: float = 0.5,
    row_nan_max: int | None = None,
    drop_constants: bool = True,
    drop_ids: bool = False,
    id_cols: List[str] | None = None,
) -> None:
    df = pd.read_csv(inp)

    dropped_cols: List[str] = []
    df, dropped = drop_high_nan_columns(df, col_nan_threshold)
    dropped_cols.extend(dropped)

    if drop_constants:
        df, dropped = drop_constant_columns(df)
        dropped_cols.extend(dropped)

    if drop_ids:
        ids = id_cols or ["seqn"]
        df, dropped = drop_id_columns(df, ids)
        dropped_cols.extend(dropped)

    if row_nan_max is not None:
        df, dropped_rows = drop_rows_by_nan_count(df, row_nan_max)
    else:
        df, dropped_rows = drop_high_nan_rows(df, row_nan_threshold)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"OK cleaned: rows={len(df)} cols={len(df.columns)}")
    if dropped_cols:
        print("INFO dropped columns:", ", ".join(sorted(set(dropped_cols))))
    if dropped_rows:
        print("INFO dropped rows:", dropped_rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Basic cleaning for NHANES CSV (NaN and constants).")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument(
        "--col-nan-threshold",
        type=float,
        default=0.95,
        help="Drop columns with NaN ratio > threshold.",
    )
    ap.add_argument(
        "--row-nan-threshold",
        type=float,
        default=0.5,
        help="Drop rows with NaN ratio > threshold.",
    )
    ap.add_argument(
        "--row-nan-max",
        type=int,
        default=None,
        help="Drop rows with more than this count of NaN values.",
    )
    ap.add_argument(
        "--drop-constants",
        action="store_true",
        help="Drop columns with a single value (including NaN).",
    )
    ap.add_argument(
        "--drop-ids",
        action="store_true",
        help="Drop ID columns (default: seqn).",
    )
    ap.add_argument(
        "--id-cols",
        default="seqn",
        help="Comma-separated ID columns to drop when --drop-ids is set.",
    )

    args = ap.parse_args()
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    clean_nhanes(
        args.inp,
        args.out,
        col_nan_threshold=float(args.col_nan_threshold),
        row_nan_threshold=float(args.row_nan_threshold),
        row_nan_max=args.row_nan_max if args.row_nan_max is not None else None,
        drop_constants=bool(args.drop_constants),
        drop_ids=bool(args.drop_ids),
        id_cols=id_cols,
    )
