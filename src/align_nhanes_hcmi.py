from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _map_sex_nhanes(series: pd.Series) -> pd.Series:
    mapping = {1: "male", 2: "female"}
    return _to_numeric(series).map(mapping)


def _map_ridreth3_to_ethnicity(series: pd.Series) -> pd.Series:
    s = _to_numeric(series)
    return s.map({1: "hispanic or latino", 2: "hispanic or latino"}).fillna("not hispanic or latino")


def _map_ridreth3_to_race(series: pd.Series) -> pd.Series:
    mapping = {
        3: "white",
        4: "black or african american",
        6: "asian",
        7: "other",
    }
    return _to_numeric(series).map(mapping)


def _smoking_any_from_nhanes(smq020: pd.Series, smq040: pd.Series) -> pd.Series:
    smq020n = _to_numeric(smq020)
    smq040n = _to_numeric(smq040)
    out = pd.Series(float("nan"), index=smq020.index, dtype="float")
    out.loc[smq020n == 1] = 1
    out.loc[smq020n == 2] = 0
    out.loc[out.isna() & smq040n.isin([1, 2])] = 1
    out.loc[out.isna() & (smq040n == 3)] = 0
    return out


def build_nhanes_aligned(
    nhanes_csv: str | Path,
    *,
    require_mcq220_no_cancer: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(nhanes_csv)

    if require_mcq220_no_cancer and "mcq220" in df.columns:
        df = df[_to_numeric(df["mcq220"]) == 2].copy()

    out = pd.DataFrame()
    out["age_years"] = _to_numeric(df.get("ridageyr"))
    out["sex"] = _map_sex_nhanes(df.get("riagendr"))
    out["ethnicity"] = _map_ridreth3_to_ethnicity(df.get("ridreth3"))
    out["race"] = _map_ridreth3_to_race(df.get("ridreth3"))
    out["height_last"] = _to_numeric(df.get("bmxht"))
    out["weight_last"] = _to_numeric(df.get("bmxwt"))
    out["bmi_last"] = _to_numeric(df.get("bmxbmi"))
    out["tobacco_smoking_status_any"] = _smoking_any_from_nhanes(
        df.get("smq020"),
        df.get("smq040"),
    )
    out["pack_years_smoked_max"] = pd.NA
    out["label"] = 0
    return out


def _age_from_hcmi(df: pd.DataFrame) -> pd.Series:
    age = pd.Series(float("nan"), index=df.index, dtype="float")

    if "dem_days_to_birth" in df.columns:
        s = _to_numeric(df["dem_days_to_birth"])
        birth_years = (-s / 365.25).round(1)
        age.loc[birth_years.notna()] = birth_years.loc[birth_years.notna()]

    if "mean_age_at_dx" in df.columns:
        s = _to_numeric(df["mean_age_at_dx"])
        dx_years = (s / 365.25).round(1)
        age.loc[age.isna() & dx_years.notna()] = dx_years.loc[age.isna() & dx_years.notna()]

    return age


def build_hcmi_aligned(hcmi_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(hcmi_csv)

    out = pd.DataFrame()
    out["age_years"] = _age_from_hcmi(df)
    out["sex"] = df.get("sex", df.get("dem_gender"))
    out["ethnicity"] = df.get("ethnicity", df.get("dem_ethnicity"))
    out["race"] = df.get("dem_race")
    out["height_last"] = _to_numeric(df.get("height_last"))
    out["weight_last"] = _to_numeric(df.get("weight_last"))
    out["bmi_last"] = _to_numeric(df.get("bmi_last"))
    out["tobacco_smoking_status_any"] = _to_numeric(df.get("tobacco_smoking_status_any"))
    out["pack_years_smoked_max"] = _to_numeric(df.get("pack_years_smoked_max"))
    out["label"] = 1
    return out


def drop_rows_by_nan_ratio(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    mask = df.isna().mean(axis=1) <= threshold
    return df.loc[mask].copy()


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Align NHANES and HCMI into a common feature set.")
    ap.add_argument("--nhanes", default="data/nhanes/nhanes_merged.csv", help="NHANES merged CSV.")
    ap.add_argument("--hcmi", default="data/training/raw.csv", help="HCMI flat CSV (from build_raw).")
    ap.add_argument("--out-nhanes", default="data/aligned/nhanes_aligned.csv", help="NHANES aligned output.")
    ap.add_argument("--out-hcmi", default="data/aligned/hcmi_aligned.csv", help="HCMI aligned output.")
    ap.add_argument("--out-combined", default="data/aligned/combined_aligned.csv", help="Combined output.")
    ap.add_argument(
        "--row-nan-threshold",
        type=float,
        default=None,
        help="Optional max NaN ratio per row (applies after alignment).",
    )

    args = ap.parse_args()

    nhanes_aligned = build_nhanes_aligned(args.nhanes, require_mcq220_no_cancer=True)
    hcmi_aligned = build_hcmi_aligned(args.hcmi)

    if args.row_nan_threshold is not None:
        nhanes_aligned = drop_rows_by_nan_ratio(nhanes_aligned, float(args.row_nan_threshold))
        hcmi_aligned = drop_rows_by_nan_ratio(hcmi_aligned, float(args.row_nan_threshold))

    save_csv(nhanes_aligned, args.out_nhanes)
    save_csv(hcmi_aligned, args.out_hcmi)

    combined = pd.concat([nhanes_aligned, hcmi_aligned], ignore_index=True)
    save_csv(combined, args.out_combined)

    print(f"OK NHANES aligned: rows={len(nhanes_aligned)} cols={len(nhanes_aligned.columns)}")
    print(f"OK HCMI aligned: rows={len(hcmi_aligned)} cols={len(hcmi_aligned.columns)}")
    print(f"OK combined: rows={len(combined)} cols={len(combined.columns)}")
