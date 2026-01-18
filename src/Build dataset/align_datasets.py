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

MISSING_TOKENS = {
    "",
    "nan",
    "none",
    "null",
    "unknown",
    "unk",
    "not reported",
    "not known",
    "not specified",
    "not applicable",
    "n/a",
    "missing",
}


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


def _normalize_feature_list(features: Optional[List[str]]) -> List[str]:
    if not features:
        return []
    return [_to_snake(str(f)) for f in features if str(f).strip()]


def _apply_feature_blindness(
    df: pd.DataFrame,
    drop_features: List[str],
    mode: str,
) -> pd.DataFrame:
    if not drop_features:
        return df
    df2 = df.copy()
    if mode == "mask":
        for col in drop_features:
            if col in df2.columns:
                df2[col] = np.nan
        return df2
    return df2.drop(columns=[c for c in drop_features if c in df2.columns], errors="ignore")


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
    s = s.replace({k: np.nan for k in MISSING_TOKENS})
    return s


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").isna()
    s = series.astype(str).str.strip().str.lower()
    return series.isna() | s.isin(MISSING_TOKENS)


def _missing_stats(series: pd.Series) -> Tuple[int, float]:
    mask = _missing_mask(series)
    missing = int(mask.sum())
    pct = float(missing / len(series)) if len(series) else 0.0
    return missing, pct


def _diagnostic_rows(
    raw_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
    dataset_name: str,
    protected_cols: List[str],
    null_checks: Dict[str, List[str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    raw_rows = len(raw_df)
    aligned_rows = len(aligned_df)

    rows.append(
        {
            "dataset": dataset_name,
            "stage": "raw_load",
            "column": "",
            "rows_total": raw_rows,
            "rows_after": raw_rows,
            "rows_lost": 0,
            "missing_count": "",
            "missing_pct": "",
            "reason": "load_csv",
            "source_column": "",
        }
    )
    rows.append(
        {
            "dataset": dataset_name,
            "stage": "after_align_filter",
            "column": "",
            "rows_total": raw_rows,
            "rows_after": aligned_rows,
            "rows_lost": int(raw_rows - aligned_rows),
            "missing_count": "",
            "missing_pct": "",
            "reason": "align_and_reindex",
            "source_column": "",
        }
    )
    rows.append(
        {
            "dataset": dataset_name,
            "stage": "after_normalize_categories",
            "column": "",
            "rows_total": aligned_rows,
            "rows_after": aligned_rows,
            "rows_lost": 0,
            "missing_count": "",
            "missing_pct": "",
            "reason": "normalize_categories",
            "source_column": "",
        }
    )

    protected_present = [c for c in protected_cols if c in aligned_df.columns]
    if protected_present:
        after_dropna = int(len(aligned_df.dropna(subset=protected_present, how="any")))
    else:
        after_dropna = aligned_rows
    rows.append(
        {
            "dataset": dataset_name,
            "stage": "after_dropna_protected",
            "column": ",".join(protected_present),
            "rows_total": aligned_rows,
            "rows_after": after_dropna,
            "rows_lost": int(aligned_rows - after_dropna),
            "missing_count": "",
            "missing_pct": "",
            "reason": "dropna_any_protected",
            "source_column": "",
        }
    )

    rows.append(
        {
            "dataset": dataset_name,
            "stage": "merge_audit",
            "column": "",
            "rows_total": raw_rows,
            "rows_after": raw_rows,
            "rows_lost": 0,
            "missing_count": "",
            "missing_pct": "",
            "reason": "no_merge_in_align",
            "source_column": "",
        }
    )

    for logical_name, candidates in null_checks.items():
        src = _pick_first(raw_df, candidates)
        if src is None or src not in raw_df.columns:
            rows.append(
                {
                    "dataset": dataset_name,
                    "stage": "null_analysis",
                    "column": logical_name,
                    "rows_total": raw_rows,
                    "rows_after": "",
                    "rows_lost": "",
                    "missing_count": "",
                    "missing_pct": "",
                    "reason": "column_missing_in_source",
                    "source_column": "",
                }
            )
            continue
        missing_count, missing_pct = _missing_stats(raw_df[src])
        rows.append(
            {
                "dataset": dataset_name,
                "stage": "null_analysis",
                "column": logical_name,
                "rows_total": raw_rows,
                "rows_after": "",
                "rows_lost": "",
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 4),
                "reason": "missing_or_not_reported",
                "source_column": src,
            }
        )
    return rows


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


def align_datasets(
    hcmi_path: Path,
    nhanes_path: Path,
    outdir: Path,
    seed: int,
    drop_features: Optional[List[str]] = None,
    blind_mode: str = "drop",
    report_loss: bool = False,
    report_path: Optional[Path] = None,
) -> None:
    hcmi_raw = _normalize_columns(pd.read_csv(hcmi_path))
    nhanes_raw = _normalize_columns(pd.read_csv(nhanes_path))

    hcmi_aligned, hcmi_map, hcmi_created = _align_hcmi(hcmi_raw)
    nhanes_aligned, nhanes_map, nhanes_created = _align_nhanes(nhanes_raw)

    drop_features_norm = _normalize_feature_list(drop_features)
    dropped_for_leakage: List[str] = []
    masked_for_leakage: List[str] = []
    if drop_features_norm:
        if blind_mode == "mask":
            masked_for_leakage = drop_features_norm
        else:
            dropped_for_leakage = drop_features_norm

        for feat in drop_features_norm:
            if feat in hcmi_map:
                hcmi_map[feat] = [f"{blind_mode}_for_leakage"]
            if feat in nhanes_map:
                nhanes_map[feat] = [f"{blind_mode}_for_leakage"]
        if blind_mode == "drop":
            hcmi_created = [c for c in hcmi_created if c not in drop_features_norm]
            nhanes_created = [c for c in nhanes_created if c not in drop_features_norm]

    # Canonical order
    if blind_mode == "mask":
        canonical_numeric = list(CANONICAL_NUMERIC)
        canonical_categorical = list(CANONICAL_CATEGORICAL)
    else:
        canonical_numeric = [c for c in CANONICAL_NUMERIC if c not in drop_features_norm]
        canonical_categorical = [c for c in CANONICAL_CATEGORICAL if c not in drop_features_norm]
    feature_cols = canonical_numeric + canonical_categorical
    ordered_cols = feature_cols + [LABEL_COL, DOMAIN_COL]
    hcmi_aligned = _apply_feature_blindness(hcmi_aligned, drop_features_norm, blind_mode)
    nhanes_aligned = _apply_feature_blindness(nhanes_aligned, drop_features_norm, blind_mode)
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
        "numeric": canonical_numeric,
        "categorical": canonical_categorical,
        "domain_col": DOMAIN_COL,
        "label_col": LABEL_COL,
        "dropped_for_leakage": dropped_for_leakage,
        "masked_for_leakage": masked_for_leakage,
        "blind_mode": blind_mode,
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

    if report_loss:
        protected_cols = [
            "age_years",
            "bmi_last",
            "height_last",
            "weight_last",
            "sex",
            "race",
            "ethnicity",
        ]
        null_checks = {
            "age_at_index": [
                "age_at_index",
                "dem_age_at_index",
                "age_years",
                "dem_days_to_birth",
                "mean_age_at_dx",
            ],
            "tobacco_smoking_status_any": ["tobacco_smoking_status_any", "smq020", "smq040"],
            "bmi_last": ["bmi_last", "bmxbmi"],
            "gender": ["gender", "sex", "dem_gender", "dem_sex_at_birth"],
            "race": ["race", "dem_race"],
        }
        report_rows: List[Dict[str, object]] = []
        report_rows.extend(
            _diagnostic_rows(
                hcmi_raw,
                hcmi_aligned,
                dataset_name="hcmi_tcga",
                protected_cols=protected_cols,
                null_checks=null_checks,
            )
        )
        report_rows.extend(
            _diagnostic_rows(
                nhanes_raw,
                nhanes_aligned,
                dataset_name="nhanes",
                protected_cols=protected_cols,
                null_checks=null_checks,
            )
        )
        report_df = pd.DataFrame(report_rows)
        out_path = report_path or (outdir / "data_loss_report.csv")
        report_df.to_csv(out_path, index=False)
        print("Loss report saved:", out_path)

        def _stage_rows(df: pd.DataFrame, dataset: str, stage: str) -> Optional[int]:
            sub = df[(df["dataset"] == dataset) & (df["stage"] == stage)]
            if sub.empty:
                return None
            return int(sub["rows_after"].iloc[0])

        for dataset in ("hcmi_tcga", "nhanes"):
            raw = _stage_rows(report_df, dataset, "raw_load")
            aligned = _stage_rows(report_df, dataset, "after_align_filter")
            norm = _stage_rows(report_df, dataset, "after_normalize_categories")
            dropna = _stage_rows(report_df, dataset, "after_dropna_protected")
            print(
                f"LOSS {dataset}:"
                f" raw={raw}"
                f" aligned={aligned}"
                f" normalized={norm}"
                f" dropna_protected={dropna}"
            )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Align HCMI/TCGA and NHANES into a canonical schema.")
    ap.add_argument("--hcmi", required=True, help="Path to HCMI/TCGA labeled CSV.")
    ap.add_argument("--nhanes", required=True, help="Path to NHANES labeled CSV.")
    ap.add_argument("--outdir", default="data/processed", help="Output directory for aligned data.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    ap.add_argument(
        "--drop-features",
        default="",
        help="Lista separada por coma de features a excluir por fuga (ej: tobacco_smoking_status_any).",
    )
    ap.add_argument(
        "--blind-mode",
        choices=["drop", "mask"],
        default="drop",
        help="Modo de ceguera de variable: drop elimina columnas, mask las pone en NaN.",
    )
    ap.add_argument(
        "--report-loss",
        action="store_true",
        help="Genera un reporte de descarte y nulos por columna.",
    )
    ap.add_argument(
        "--report-path",
        default="",
        help="Ruta opcional para data_loss_report.csv (por defecto en outdir).",
    )
    args = ap.parse_args()

    drop_features = [c.strip() for c in args.drop_features.split(",") if c.strip()]
    align_datasets(
        Path(args.hcmi),
        Path(args.nhanes),
        Path(args.outdir),
        seed=int(args.seed),
        drop_features=drop_features,
        blind_mode=str(args.blind_mode),
        report_loss=bool(args.report_loss),
        report_path=Path(args.report_path) if args.report_path else None,
    )
