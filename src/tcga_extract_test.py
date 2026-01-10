from __future__ import annotations
"""
Extract TCGA clinical JSON into the aligned feature set used for training.

Outputs a CSV next to the input JSON:
  <input_stem>_aligned.csv
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "age_years",
    "sex",
    "ethnicity",
    "race",
    "height_last",
    "weight_last",
    "bmi_last",
    "tobacco_smoking_status_any",
    "pack_years_smoked_max",
    "label",
]


def _to_numeric(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _normalize_cat(v) -> str | float:
    if v is None:
        return np.nan
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "not reported", "unknown"):
        return np.nan
    return s


def _age_years(demo: Dict[str, Any]) -> float:
    age_at_index = demo.get("age_at_index")
    if age_at_index is not None:
        age = _to_numeric(age_at_index)
        if 0 < age <= 120:
            return age
    days_to_birth = demo.get("days_to_birth")
    if days_to_birth is not None:
        days = _to_numeric(days_to_birth)
        if not np.isnan(days):
            return round(-days / 365.25, 1)
    return float("nan")


def _extract_rows(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        demo = rec.get("demographic") or {}
        row = {
            "age_years": _age_years(demo),
            "sex": _normalize_cat(demo.get("gender") or demo.get("sex_at_birth")),
            "ethnicity": _normalize_cat(demo.get("ethnicity")),
            "race": _normalize_cat(demo.get("race")),
            "height_last": np.nan,
            "weight_last": np.nan,
            "bmi_last": np.nan,
            "tobacco_smoking_status_any": np.nan,
            "pack_years_smoked_max": np.nan,
            "label": 1,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLS)


def extract_tcga_json(input_path: Path) -> Path:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = data.get("data") or data.get("records") or []
        if not isinstance(records, list):
            records = [data]
    else:
        records = data

    df = _extract_rows(records)
    out_path = input_path.with_name(f"{input_path.stem}_aligned.csv")
    df.to_csv(out_path, index=False)
    print(f"OK TCGA aligned: rows={len(df)} cols={len(df.columns)} -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract TCGA clinical JSON to aligned CSV.")
    ap.add_argument("--input", required=True, help="Path to TCGA clinical JSON.")
    args = ap.parse_args()

    extract_tcga_json(Path(args.input))
