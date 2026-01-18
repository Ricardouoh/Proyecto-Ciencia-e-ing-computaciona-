from __future__ import annotations
"""
Re-extract clinical tables from JSON files and keep only useful tables.

Default behavior:
- Reads each *.json/*.jsonl/*.ndjson in input dir.
- Extracts tables using hcmi_loader.flatten_hcmi.
- Keeps tables that contain training-relevant columns.
- Writes per-JSON subfolders with only useful CSVs.
- Produces a summary report (tables_report.csv).
"""

from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

from src.hcmi_loader import flatten_hcmi, load_hcmi


KEEP_COLUMNS = {
    "age_years",
    "age_at_index",
    "dem_age_at_index",
    "dem_days_to_birth",
    "mean_age_at_dx",
    "age_at_diagnosis",
    "sex",
    "gender",
    "dem_gender",
    "dem_sex_at_birth",
    "race",
    "dem_race",
    "ethnicity",
    "dem_ethnicity",
    "tobacco_smoking_status_any",
    "tobacco_smoking_status",
    "smq020",
    "smq040",
    "pack_years_smoked",
    "pack_years_smoked_max",
    "bmi",
    "bmi_last",
    "bmxbmi",
    "height",
    "height_last",
    "bmxht",
    "weight",
    "weight_last",
    "bmxwt",
}


def _normalize_cols(cols: Iterable[str]) -> Set[str]:
    return {str(c).strip().lower() for c in cols}


def _useful_table(name: str, df: pd.DataFrame, keep_cols: Set[str]) -> List[str]:
    if df is None or df.empty:
        return []
    cols = _normalize_cols(df.columns)
    hits = sorted(cols & keep_cols)
    return hits


def _iter_json_files(input_dir: Path) -> List[Path]:
    patterns = ("*.json", "*.jsonl", "*.ndjson")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(input_dir.glob(pattern)))
    return files


def reextract(
    input_dir: Path,
    outdir: Path,
    keep_cols: Set[str],
    reset_output: bool,
) -> None:
    input_dir = input_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    report_rows: List[Dict[str, object]] = []
    json_files = _iter_json_files(input_dir)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    for json_path in json_files:
        project = json_path.stem
        project_dir = outdir / project
        if reset_output and project_dir.exists():
            for old_csv in project_dir.glob("*.csv"):
                old_csv.unlink()
        project_dir.mkdir(parents=True, exist_ok=True)

        records = load_hcmi(json_path)
        tables = flatten_hcmi(records)

        for table_name, df in tables.items():
            hits = _useful_table(table_name, df, keep_cols)
            keep = bool(hits)
            reason = "kept" if keep else "no_training_columns"
            rows = 0 if df is None else len(df)

            report_rows.append(
                {
                    "dataset": project,
                    "table": table_name,
                    "rows": rows,
                    "kept": int(keep),
                    "matched_columns": ",".join(hits),
                    "reason": reason,
                }
            )
            if keep and df is not None and not df.empty:
                out_path = project_dir / f"{table_name}.csv"
                df.to_csv(out_path, index=False)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(outdir / "tables_report.csv", index=False)
    print("OK report:", outdir / "tables_report.csv")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Re-extract clinical tables and keep only useful CSVs.")
    ap.add_argument("--input-dir", required=True, help="Directory with JSON files.")
    ap.add_argument("--outdir", required=True, help="Output directory for cleaned tables.")
    ap.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete existing CSVs inside each project folder before writing new ones.",
    )
    args = ap.parse_args()

    reextract(
        input_dir=Path(args.input_dir),
        outdir=Path(args.outdir),
        keep_cols=set(KEEP_COLUMNS),
        reset_output=bool(args.reset_output),
    )
