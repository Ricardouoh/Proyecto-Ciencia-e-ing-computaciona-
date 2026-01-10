from __future__ import annotations
"""
Download and align NHANES 2001-2002 data to the model feature set.

Outputs (in outdir):
- raw/*.XPT (downloaded modules)
- nhanes_2001_2002_merged.csv
- nhanes_2001_2002_aligned.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request

import numpy as np
import pandas as pd


MODULES_DEFAULT = ["DEMO", "BMX", "SMQ", "MCQ"]
SUFFIX_BY_BEGIN_YEAR = {
    1999: "A",
    2001: "B",
    2003: "C",
    2005: "D",
    2007: "E",
    2009: "F",
    2011: "G",
    2013: "H",
    2015: "I",
    2017: "J",
}


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _map_sex(series: pd.Series) -> pd.Series:
    mapping = {1: "male", 2: "female"}
    return _to_numeric(series).map(mapping)


def _map_ethnicity_race(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if "ridreth3" in df.columns:
        s = _to_numeric(df["ridreth3"])
        ethnicity = s.map({1: "hispanic or latino", 2: "hispanic or latino"}).fillna(
            "not hispanic or latino"
        )
        race = s.map(
            {
                3: "white",
                4: "black or african american",
                6: "asian",
                7: "other",
            }
        )
        return ethnicity, race

    if "ridreth1" in df.columns:
        s = _to_numeric(df["ridreth1"])
        ethnicity = s.map({1: "hispanic or latino", 2: "hispanic or latino"}).fillna(
            "not hispanic or latino"
        )
        race = s.map(
            {
                3: "white",
                4: "black or african american",
                5: "other",
                1: "other",
                2: "other",
            }
        )
        return ethnicity, race

    return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)


def _smoking_any_from_nhanes(smq020: pd.Series, smq040: pd.Series) -> pd.Series:
    smq020n = _to_numeric(smq020)
    smq040n = _to_numeric(smq040)
    out = pd.Series(float("nan"), index=smq020.index, dtype="float")
    out.loc[smq020n == 1] = 1
    out.loc[smq020n == 2] = 0
    out.loc[out.isna() & smq040n.isin([1, 2])] = 1
    out.loc[out.isna() & (smq040n == 3)] = 0
    return out


def _download_files(
    base_url: str,
    outdir: Path,
    modules: List[str],
    *,
    force: bool = False,
) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for mod in modules:
        url = f"{base_url}/{mod}.XPT"
        dest = outdir / f"{mod}.XPT"
        if force or not dest.exists():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, dest)
        paths[mod] = dest
    return paths


def _read_xpt(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        head = f.read(64)
    if b"HEADER RECORD" not in head:
        raise ValueError(
            f"{path.name} no parece un XPT valido. "
            "Descarga manualmente los XPT desde NHANES y guardalos en el folder raw."
        )
    df = pd.read_sas(path, format="xport")
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _merge_by_seqn(paths: Dict[str, Path]) -> pd.DataFrame:
    frames = []
    for mod, path in paths.items():
        df = _read_xpt(path)
        if "seqn" not in df.columns:
            raise ValueError(f"Falta SEQN en {mod}")
        frames.append(df)

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="seqn", how="outer", suffixes=("", "_dup"))
    return merged


def _align_nhanes(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["age_years"] = _to_numeric(df.get("ridageyr"))
    out["sex"] = _map_sex(df.get("riagendr"))
    ethnicity, race = _map_ethnicity_race(df)
    out["ethnicity"] = ethnicity
    out["race"] = race
    out["height_last"] = _to_numeric(df.get("bmxht"))
    out["weight_last"] = _to_numeric(df.get("bmxwt"))
    out["bmi_last"] = _to_numeric(df.get("bmxbmi"))
    out["tobacco_smoking_status_any"] = _smoking_any_from_nhanes(
        df.get("smq020"), df.get("smq040")
    )
    out["pack_years_smoked_max"] = np.nan
    out["label"] = 0
    return out


def extract_nhanes_2001(
    outdir: Path,
    require_no_cancer: bool,
    *,
    skip_download: bool = False,
    force_download: bool = False,
) -> None:
    begin_year = 2001
    cycle = f"{begin_year}-{begin_year + 1}"
    suffix = SUFFIX_BY_BEGIN_YEAR[begin_year]

    # Use public data files endpoint (direct XPTs).
    base_url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{begin_year}/DataFiles"
    modules = [f"{m}_{suffix}" for m in MODULES_DEFAULT]

    raw_dir = outdir / "raw"
    if skip_download:
        paths = {mod: raw_dir / f"{mod}.XPT" for mod in modules}
    else:
        paths = _download_files(base_url, raw_dir, modules, force=force_download)

    merged = _merge_by_seqn(paths)
    if require_no_cancer and "mcq220" in merged.columns:
        merged = merged[_to_numeric(merged["mcq220"]) == 2].copy()

    outdir.mkdir(parents=True, exist_ok=True)
    merged_path = outdir / "nhanes_2001_2002_merged.csv"
    merged.to_csv(merged_path, index=False)

    aligned = _align_nhanes(merged)
    aligned_path = outdir / "nhanes_2001_2002_aligned.csv"
    aligned.to_csv(aligned_path, index=False)

    print(f"OK merged rows={len(merged)} cols={len(merged.columns)} -> {merged_path}")
    print(f"OK aligned rows={len(aligned)} cols={len(aligned.columns)} -> {aligned_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download + align NHANES 2001-2002 for testing.")
    ap.add_argument(
        "--outdir",
        default="data/tests/nhanes_2001_2002_test",
        help="Output directory for merged/aligned files.",
    )
    ap.add_argument(
        "--allow-cancer",
        action="store_true",
        help="Do not filter MCQ220==2 (keep all rows).",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloads and use existing XPTs in outdir/raw.",
    )
    ap.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files exist.",
    )
    args = ap.parse_args()

    extract_nhanes_2001(
        Path(args.outdir),
        require_no_cancer=not bool(args.allow_cancer),
        skip_download=bool(args.skip_download),
        force_download=bool(args.force_download),
    )
