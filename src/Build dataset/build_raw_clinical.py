from __future__ import annotations
"""
Build a raw clinical dataset from re-extracted project tables.

Expected input layout:
  data/clinical/reextracted/<project>/cases.csv
  data/clinical/reextracted/<project>/diagnoses.csv (optional)
  data/clinical/reextracted/<project>/exposures.csv (optional)
  data/clinical/reextracted/<project>/other_clinical_attributes.csv (optional)
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_category(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    s = s.replace({k: np.nan for k in MISSING_TOKENS})
    return s


def _age_years_from_cases(
    cases: pd.DataFrame,
    diag_age_years: Optional[pd.Series] = None,
) -> pd.Series:
    age = pd.Series(np.nan, index=cases.index, dtype="float")
    if "dem_age_at_index" in cases.columns:
        age = _to_numeric(cases["dem_age_at_index"])
    if "dem_days_to_birth" in cases.columns:
        birth_years = (-_to_numeric(cases["dem_days_to_birth"]) / 365.25).round(1)
        age = age.fillna(birth_years)
    if diag_age_years is not None:
        age = age.fillna(diag_age_years)
    return age


def _sex_from_cases(cases: pd.DataFrame) -> pd.Series:
    if "dem_gender" in cases.columns:
        return _clean_category(cases["dem_gender"])
    if "dem_sex_at_birth" in cases.columns:
        return _clean_category(cases["dem_sex_at_birth"])
    return pd.Series(np.nan, index=cases.index)


def _race_from_cases(cases: pd.DataFrame) -> pd.Series:
    if "dem_race" in cases.columns:
        return _clean_category(cases["dem_race"])
    return pd.Series(np.nan, index=cases.index)


def _ethnicity_from_cases(cases: pd.DataFrame) -> pd.Series:
    if "dem_ethnicity" in cases.columns:
        return _clean_category(cases["dem_ethnicity"])
    return pd.Series(np.nan, index=cases.index)


def _map_smoking_status(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        if value == 0:
            return 0
        if value == 1:
            return 1
    s = str(value).strip().lower()
    if s in MISSING_TOKENS:
        return None
    if any(k in s for k in ("current", "former", "ever", "smok")):
        return 1
    if any(k in s for k in ("never", "no", "non")):
        return 0
    if s in {"true", "t", "yes", "y", "1"}:
        return 1
    if s in {"false", "f", "0"}:
        return 0
    return None


def _aggregate_exposures(exposures: pd.DataFrame) -> pd.DataFrame:
    if exposures is None or exposures.empty:
        return pd.DataFrame(columns=["case_id", "tobacco_smoking_status_any", "pack_years_smoked_max"])
    exp = exposures.copy()
    exp["tobacco_smoking_status_any"] = exp.get("tobacco_smoking_status").map(_map_smoking_status)
    exp["pack_years_smoked"] = _to_numeric(exp.get("pack_years_smoked"))

    def _status_any(vals: pd.Series) -> Optional[int]:
        v = vals.dropna().astype(int)
        if v.empty:
            return None
        if (v == 1).any():
            return 1
        if (v == 0).any():
            return 0
        return None

    grouped = exp.groupby("case_id", dropna=True)
    status_any = grouped["tobacco_smoking_status_any"].apply(_status_any)
    pack_max = grouped["pack_years_smoked"].max()

    out = pd.DataFrame(
        {
            "case_id": status_any.index,
            "tobacco_smoking_status_any": status_any.values,
            "pack_years_smoked_max": pack_max.reindex(status_any.index).values,
        }
    )
    return out


def _aggregate_other_attrs(other: pd.DataFrame) -> pd.DataFrame:
    if other is None or other.empty:
        return pd.DataFrame(columns=["case_id", "bmi_last", "height_last", "weight_last"])
    oa = other.copy()
    if "updated_datetime" in oa.columns:
        oa["_dt"] = pd.to_datetime(oa["updated_datetime"], errors="coerce")
    else:
        oa["_dt"] = pd.NaT

    for col in ("bmi", "height", "weight"):
        if col in oa.columns:
            oa[col] = _to_numeric(oa[col])

    def _last_non_null(group: pd.DataFrame, col: str) -> float:
        g = group.sort_values("_dt")
        vals = g[col].dropna()
        if not vals.empty:
            return float(vals.iloc[-1])
        return float("nan")

    grouped = oa.groupby("case_id", dropna=True)
    bmi_last = grouped.apply(lambda g: _last_non_null(g, "bmi"))
    height_last = grouped.apply(lambda g: _last_non_null(g, "height"))
    weight_last = grouped.apply(lambda g: _last_non_null(g, "weight"))

    out = pd.DataFrame(
        {
            "case_id": bmi_last.index,
            "bmi_last": bmi_last.values,
            "height_last": height_last.reindex(bmi_last.index).values,
            "weight_last": weight_last.reindex(bmi_last.index).values,
        }
    )
    return out


def _aggregate_diagnoses(diag: pd.DataFrame) -> pd.Series:
    if diag is None or diag.empty or "age_at_diagnosis" not in diag.columns:
        return pd.Series(dtype=float)
    ages = _to_numeric(diag["age_at_diagnosis"]) / 365.25
    ages = ages.round(1)
    out = diag.assign(age_years=ages).groupby("case_id")["age_years"].median()
    return out


def build_raw_dataset(
    input_dir: Path,
    outdir: Path,
    label_value: int = 1,
    domain_value: str = "hcmi_tcga",
    per_project: bool = True,
) -> None:
    input_dir = input_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    project_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    report_rows: List[Dict[str, object]] = []
    combined_rows: List[pd.DataFrame] = []

    for proj_dir in sorted(project_dirs):
        cases_path = proj_dir / "cases.csv"
        if not cases_path.exists():
            continue

        cases = pd.read_csv(cases_path)
        if "case_id" not in cases.columns:
            continue

        diag = pd.read_csv(proj_dir / "diagnoses.csv") if (proj_dir / "diagnoses.csv").exists() else pd.DataFrame()
        exposures = pd.read_csv(proj_dir / "exposures.csv") if (proj_dir / "exposures.csv").exists() else pd.DataFrame()
        other = (
            pd.read_csv(proj_dir / "other_clinical_attributes.csv")
            if (proj_dir / "other_clinical_attributes.csv").exists()
            else pd.DataFrame()
        )

        diag_age = _aggregate_diagnoses(diag)
        age_years = _age_years_from_cases(cases, diag_age)
        sex = _sex_from_cases(cases)
        race = _race_from_cases(cases)
        eth = _ethnicity_from_cases(cases)

        exp_agg = _aggregate_exposures(exposures)
        other_agg = _aggregate_other_attrs(other)

        base = pd.DataFrame(
            {
                "case_id": cases["case_id"].astype(str),
                "case_submitter_id": cases.get("case_submitter_id", pd.Series(np.nan, index=cases.index)).astype(str),
                "source_project": proj_dir.name,
                "age_years": age_years,
                "sex": sex,
                "ethnicity": eth,
                "race": race,
            }
        )

        merged = base.merge(exp_agg, on="case_id", how="left")
        merged = merged.merge(other_agg, on="case_id", how="left")

        merged["label"] = int(label_value)
        merged["domain"] = str(domain_value)

        merged = merged[
            [
                "case_id",
                "case_submitter_id",
                "source_project",
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
                "domain",
            ]
        ]

        if per_project:
            out_path = outdir / f"{proj_dir.name}_raw.csv"
            merged.to_csv(out_path, index=False)

        combined_rows.append(merged)
        report_rows.append(
            {
                "project": proj_dir.name,
                "rows": len(merged),
                "age_missing_pct": float(merged["age_years"].isna().mean()) if len(merged) else 0.0,
                "bmi_missing_pct": float(merged["bmi_last"].isna().mean()) if len(merged) else 0.0,
                "height_missing_pct": float(merged["height_last"].isna().mean()) if len(merged) else 0.0,
                "weight_missing_pct": float(merged["weight_last"].isna().mean()) if len(merged) else 0.0,
                "race_missing_pct": float(merged["race"].isna().mean()) if len(merged) else 0.0,
            }
        )

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined.to_csv(outdir / "raw_hcmi_tcga.csv", index=False)

    if report_rows:
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(outdir / "raw_build_report.csv", index=False)

    print("OK raw dataset:", outdir / "raw_hcmi_tcga.csv")
    print("OK report:", outdir / "raw_build_report.csv")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build raw clinical dataset from re-extracted tables.")
    ap.add_argument("--input-dir", default="data/clinical/reextracted", help="Base folder with project subfolders.")
    ap.add_argument("--outdir", default="data/clinical/reextracted/raw", help="Output folder for raw CSVs.")
    ap.add_argument("--label", type=int, default=1, help="Label value for all rows.")
    ap.add_argument("--domain", default="hcmi_tcga", help="Domain label for all rows.")
    ap.add_argument("--no-per-project", action="store_true", help="Do not write per-project CSVs.")
    args = ap.parse_args()

    build_raw_dataset(
        input_dir=Path(args.input_dir),
        outdir=Path(args.outdir),
        label_value=int(args.label),
        domain_value=str(args.domain),
        per_project=not bool(args.no_per_project),
    )
