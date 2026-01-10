from __future__ import annotations
"""
Align two domains (HCMI/TCGA and NHANES) into a canonical schema.

Outputs:
- A_aligned.csv (hcmi/tcga)
- B_aligned.csv (nhanes)
- aligned_train.csv / aligned_val.csv / aligned_test.csv
- schema.json
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CANONICAL_NUMERIC = [
    "age_years",
    "height_last",
    "weight_last",
    "bmi_last",
    "tobacco_smoking_status_any",
    "pack_years_smoked_max",
]

CANONICAL_CATEGORICAL = [
    "sex",
    "ethnicity",
    "race",
]

DOMAIN_COL = "domain"
LABEL_COL = "label"


def _to_snake(name: str) -> str:
    return (
        name.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("__", "_")
        .lower()
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_to_snake(str(c)) for c in df.columns]
    return df


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _age_from_hcmi(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    used: List[str] = []
    if "age_years" in df.columns:
        used.append("age_years")
        return _to_numeric(df["age_years"]), used

    age = pd.Series(float("nan"), index=df.index, dtype="float")
    if "dem_days_to_birth" in df.columns:
        used.append("dem_days_to_birth")
        s = _to_numeric(df["dem_days_to_birth"])
        birth_years = (-s / 365.25).round(1)
        age.loc[birth_years.notna()] = birth_years.loc[birth_years.notna()]

    if "mean_age_at_dx" in df.columns:
        used.append("mean_age_at_dx")
        s = _to_numeric(df["mean_age_at_dx"])
        dx_years = (s / 365.25).round(1)
        age.loc[age.isna() & dx_years.notna()] = dx_years.loc[age.isna() & dx_years.notna()]

    return age, used


def _map_nhanes_sex(series: pd.Series) -> pd.Series:
    mapping = {1: "male", 2: "female"}
    s = _to_numeric(series)
    return s.map(mapping)


def _map_nhanes_ethnicity(series: pd.Series) -> pd.Series:
    s = _to_numeric(series)
    return s.map({1: "hispanic or latino", 2: "hispanic or latino"}).fillna("not hispanic or latino")


def _map_nhanes_race(series: pd.Series) -> pd.Series:
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


def _normalize_category(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    s = s.replace({"": np.nan, "nan": np.nan, "none": np.nan})
    return s


def _align_hcmi(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    mapping: Dict[str, List[str]] = {}
    created: List[str] = []
    out = pd.DataFrame(index=df.index)

    age, used = _age_from_hcmi(df)
    out["age_years"] = age
    mapping["age_years"] = used or ["created_nan"]

    sex_col = _pick_first(df, ["sex", "dem_gender"])
    if sex_col:
        out["sex"] = _normalize_category(df[sex_col])
        mapping["sex"] = [sex_col]
    else:
        out["sex"] = np.nan
        mapping["sex"] = ["created_nan"]
        created.append("sex")

    eth_col = _pick_first(df, ["ethnicity", "dem_ethnicity"])
    if eth_col:
        out["ethnicity"] = _normalize_category(df[eth_col])
        mapping["ethnicity"] = [eth_col]
    else:
        out["ethnicity"] = np.nan
        mapping["ethnicity"] = ["created_nan"]
        created.append("ethnicity")

    race_col = _pick_first(df, ["race", "dem_race"])
    if race_col:
        out["race"] = _normalize_category(df[race_col])
        mapping["race"] = [race_col]
    else:
        out["race"] = np.nan
        mapping["race"] = ["created_nan"]
        created.append("race")

    for col in ["height_last", "weight_last", "bmi_last", "tobacco_smoking_status_any", "pack_years_smoked_max"]:
        if col in df.columns:
            out[col] = _to_numeric(df[col])
            mapping[col] = [col]
        else:
            out[col] = np.nan
            mapping[col] = ["created_nan"]
            created.append(col)

    if LABEL_COL in df.columns:
        out[LABEL_COL] = _to_numeric(df[LABEL_COL]).fillna(1).astype(int)
        mapping[LABEL_COL] = [LABEL_COL]
    else:
        out[LABEL_COL] = 1
        mapping[LABEL_COL] = ["created_label_1"]

    out[DOMAIN_COL] = "hcmi_tcga"
    return out, mapping, created


def _align_nhanes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    mapping: Dict[str, List[str]] = {}
    created: List[str] = []
    out = pd.DataFrame(index=df.index)

    age_col = _pick_first(df, ["age_years", "ridageyr"])
    if age_col == "ridageyr":
        out["age_years"] = _to_numeric(df[age_col])
    elif age_col:
        out["age_years"] = _to_numeric(df[age_col])
    else:
        out["age_years"] = np.nan
        created.append("age_years")
    mapping["age_years"] = [age_col] if age_col else ["created_nan"]

    sex_col = _pick_first(df, ["sex", "riagendr"])
    if sex_col == "riagendr":
        out["sex"] = _map_nhanes_sex(df[sex_col])
    elif sex_col:
        out["sex"] = _normalize_category(df[sex_col])
    else:
        out["sex"] = np.nan
        created.append("sex")
    mapping["sex"] = [sex_col] if sex_col else ["created_nan"]

    if "ridreth3" in df.columns:
        out["ethnicity"] = _map_nhanes_ethnicity(df["ridreth3"])
        out["race"] = _map_nhanes_race(df["ridreth3"])
        mapping["ethnicity"] = ["ridreth3"]
        mapping["race"] = ["ridreth3"]
    else:
        eth_col = _pick_first(df, ["ethnicity"])
        race_col = _pick_first(df, ["race"])
        if eth_col:
            out["ethnicity"] = _normalize_category(df[eth_col])
            mapping["ethnicity"] = [eth_col]
        else:
            out["ethnicity"] = np.nan
            mapping["ethnicity"] = ["created_nan"]
            created.append("ethnicity")
        if race_col:
            out["race"] = _normalize_category(df[race_col])
            mapping["race"] = [race_col]
        else:
            out["race"] = np.nan
            mapping["race"] = ["created_nan"]
            created.append("race")

    for col, candidates in {
        "height_last": ["height_last", "bmxht", "height"],
        "weight_last": ["weight_last", "bmxwt", "weight"],
        "bmi_last": ["bmi_last", "bmxbmi", "bmi"],
    }.items():
        src = _pick_first(df, candidates)
        if src:
            out[col] = _to_numeric(df[src])
            mapping[col] = [src]
        else:
            out[col] = np.nan
            mapping[col] = ["created_nan"]
            created.append(col)

    if "tobacco_smoking_status_any" in df.columns:
        out["tobacco_smoking_status_any"] = _to_numeric(df["tobacco_smoking_status_any"])
        mapping["tobacco_smoking_status_any"] = ["tobacco_smoking_status_any"]
    elif "smq020" in df.columns and "smq040" in df.columns:
        out["tobacco_smoking_status_any"] = _smoking_any_from_nhanes(df["smq020"], df["smq040"])
        mapping["tobacco_smoking_status_any"] = ["smq020", "smq040"]
    else:
        out["tobacco_smoking_status_any"] = np.nan
        mapping["tobacco_smoking_status_any"] = ["created_nan"]
        created.append("tobacco_smoking_status_any")

    out["pack_years_smoked_max"] = np.nan
    mapping["pack_years_smoked_max"] = ["created_nan"]
    created.append("pack_years_smoked_max")

    if LABEL_COL in df.columns:
        out[LABEL_COL] = _to_numeric(df[LABEL_COL]).fillna(0).astype(int)
        mapping[LABEL_COL] = [LABEL_COL]
    else:
        out[LABEL_COL] = 0
        mapping[LABEL_COL] = ["created_label_0"]

    out[DOMAIN_COL] = "nhanes"
    return out, mapping, created


def _split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL])
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    rel_test = 0.15 / (0.15 + 0.15)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, random_state=seed, stratify=y_tmp
    )

    train = X_train.copy()
    train[LABEL_COL] = y_train.values
    val = X_val.copy()
    val[LABEL_COL] = y_val.values
    test = X_test.copy()
    test[LABEL_COL] = y_test.values
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def align_datasets(hcmi_path: Path, nhanes_path: Path, outdir: Path, seed: int) -> None:
    hcmi_raw = _normalize_columns(pd.read_csv(hcmi_path))
    nhanes_raw = _normalize_columns(pd.read_csv(nhanes_path))

    hcmi_aligned, hcmi_map, hcmi_created = _align_hcmi(hcmi_raw)
    nhanes_aligned, nhanes_map, nhanes_created = _align_nhanes(nhanes_raw)

    # Canonical order
    feature_cols = CANONICAL_NUMERIC + CANONICAL_CATEGORICAL
    ordered_cols = feature_cols + [LABEL_COL, DOMAIN_COL]
    hcmi_aligned = hcmi_aligned.reindex(columns=ordered_cols)
    nhanes_aligned = nhanes_aligned.reindex(columns=ordered_cols)

    combined = pd.concat([hcmi_aligned, nhanes_aligned], ignore_index=True)

    outdir.mkdir(parents=True, exist_ok=True)
    hcmi_aligned.to_csv(outdir / "A_aligned.csv", index=False)
    nhanes_aligned.to_csv(outdir / "B_aligned.csv", index=False)
    combined.to_csv(outdir / "combined_aligned.csv", index=False)

    train, val, test = _split(combined, seed=seed)
    train.to_csv(outdir / "aligned_train.csv", index=False)
    val.to_csv(outdir / "aligned_val.csv", index=False)
    test.to_csv(outdir / "aligned_test.csv", index=False)

    common_cols = set(hcmi_aligned.columns) & set(nhanes_aligned.columns)

    schema = {
        "columns": ordered_cols,
        "numeric": CANONICAL_NUMERIC,
        "categorical": CANONICAL_CATEGORICAL,
        "domain_col": DOMAIN_COL,
        "label_col": LABEL_COL,
        "mappings": {
            "hcmi_tcga": hcmi_map,
            "nhanes": nhanes_map,
        },
        "created_columns": {
            "hcmi_tcga": sorted(list(set(hcmi_created))),
            "nhanes": sorted(list(set(nhanes_created))),
        },
        "common_columns_count": len(common_cols),
        "created_columns_count": {
            "hcmi_tcga": len(set(hcmi_created)),
            "nhanes": len(set(nhanes_created)),
        },
    }
    with open(outdir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print("OK aligned A rows:", len(hcmi_aligned))
    print("OK aligned B rows:", len(nhanes_aligned))
    print("OK combined rows:", len(combined))
    print("Columns common:", len(common_cols))
    print("Columns created (A):", len(schema["created_columns"]["hcmi_tcga"]))
    print("Columns created (B):", len(schema["created_columns"]["nhanes"]))
    print("Saved schema:", outdir / "schema.json")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Align HCMI/TCGA and NHANES into a canonical schema.")
    ap.add_argument("--hcmi", required=True, help="Path to HCMI/TCGA labeled CSV.")
    ap.add_argument("--nhanes", required=True, help="Path to NHANES labeled CSV.")
    ap.add_argument("--outdir", default="data/processed", help="Output directory for aligned data.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    args = ap.parse_args()

    align_datasets(Path(args.hcmi), Path(args.nhanes), Path(args.outdir), seed=int(args.seed))
