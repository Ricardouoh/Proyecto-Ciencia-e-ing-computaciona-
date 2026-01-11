from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _load_data(data_dir: Path, csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return pd.read_csv(p)

    combined = data_dir / "combined_aligned.csv"
    if combined.exists():
        return pd.read_csv(combined)

    aligned = [data_dir / n for n in ("aligned_train.csv", "aligned_val.csv", "aligned_test.csv")]
    if all(p.exists() for p in aligned):
        frames = [pd.read_csv(p) for p in aligned]
        return pd.concat(frames, ignore_index=True)

    splits = [data_dir / n for n in ("train.csv", "val.csv", "test.csv")]
    if all(p.exists() for p in splits):
        frames = [pd.read_csv(p) for p in splits]
        return pd.concat(frames, ignore_index=True)

    raise FileNotFoundError("No input data found. Provide --csv or valid data-dir.")


def _calc_2x2(y: pd.Series, mask: pd.Series) -> Tuple[int, int, int, int]:
    yv = y.astype(int)
    mv = mask.fillna(False).astype(bool)
    a = int(((yv == 1) & mv).sum())
    b = int(((yv == 0) & mv).sum())
    c = int(((yv == 1) & ~mv).sum())
    d = int(((yv == 0) & ~mv).sum())
    return a, b, c, d


def _or_rr(a: int, b: int, c: int, d: int) -> Tuple[float, float, float, float, float]:
    aa, bb, cc, dd = float(a), float(b), float(c), float(d)
    if min(aa, bb, cc, dd) == 0.0:
        aa += 0.5
        bb += 0.5
        cc += 0.5
        dd += 0.5

    odds_exposed = aa / bb if bb > 0 else np.nan
    odds_unexposed = cc / dd if dd > 0 else np.nan
    odds_ratio = odds_exposed / odds_unexposed if odds_unexposed > 0 else np.nan

    risk_exposed = aa / (aa + bb) if (aa + bb) > 0 else np.nan
    risk_unexposed = cc / (cc + dd) if (cc + dd) > 0 else np.nan
    risk_ratio = risk_exposed / risk_unexposed if risk_unexposed > 0 else np.nan
    risk_diff = risk_exposed - risk_unexposed
    return odds_ratio, risk_ratio, risk_exposed, risk_unexposed, risk_diff


def _chi2_stat(a: int, b: int, c: int, d: int) -> Optional[float]:
    n = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        return None
    return float(n * (a * d - b * c) ** 2 / denom)


def _chi2_pvalue(stat: Optional[float]) -> Optional[float]:
    if stat is None:
        return None
    try:
        from scipy.stats import chi2

        return float(chi2.sf(stat, df=1))
    except Exception:
        return None


def _add_binary_feature(
    rows: List[Dict[str, object]],
    y: pd.Series,
    mask: pd.Series,
    feature: str,
    category: str = "yes",
) -> None:
    a, b, c, d = _calc_2x2(y, mask)
    odds_ratio, risk_ratio, risk_exposed, risk_unexposed, risk_diff = _or_rr(a, b, c, d)
    stat = _chi2_stat(a, b, c, d)
    pval = _chi2_pvalue(stat)
    rows.append(
        {
            "feature": feature,
            "category": category,
            "n_total": a + b,
            "n_pos": a,
            "n_neg": b,
            "odds_ratio": odds_ratio,
            "risk_ratio": risk_ratio,
            "risk_exposed": risk_exposed,
            "risk_unexposed": risk_unexposed,
            "risk_diff": risk_diff,
            "chi2": stat,
            "p_value": pval,
        }
    )


def _add_categorical_feature(
    rows: List[Dict[str, object]],
    y: pd.Series,
    s: pd.Series,
    feature: str,
    min_count: int,
) -> None:
    vals = s.dropna().astype(str).str.strip().str.lower()
    if vals.empty:
        return
    counts = vals.value_counts()
    keep = set(counts[counts >= min_count].index.tolist())
    mapped = vals.where(vals.isin(keep), "other")
    mapped_counts = mapped.value_counts()
    for cat, cnt in mapped_counts.items():
        if cnt < min_count:
            continue
        mask = mapped == cat
        _add_binary_feature(rows, y.loc[mask.index], mask, feature=feature, category=str(cat))


def _age_bins(age: pd.Series) -> pd.Series:
    bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf]
    labels = [
        "<20",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70-79",
        "80+",
    ]
    return pd.cut(age, bins=bins, labels=labels, right=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Risk stats by group (odds ratio, risk ratio)")
    ap.add_argument("--data-dir", default="data/processed", help="Base dir for aligned splits")
    ap.add_argument("--csv", default="", help="Optional CSV path with aligned data")
    ap.add_argument("--out", default="results/eda/risk_stats.csv", help="Output CSV path")
    ap.add_argument("--min-count", type=int, default=100, help="Min group count to include")
    ap.add_argument("--risk-min-age", type=float, default=60.0, help="Min age for risk group")
    ap.add_argument("--risk-use-race", action="store_true", help="Include race=white in risk group")
    args = ap.parse_args()

    df = _load_data(Path(args.data_dir), args.csv)
    if "label" not in df.columns:
        raise ValueError("Missing label column in input data")

    y = df["label"].astype(int)
    rows: List[Dict[str, object]] = []

    # Age bins
    if "age_years" in df.columns:
        age = pd.to_numeric(df["age_years"], errors="coerce")
        age_bin = _age_bins(age)
        for cat in age_bin.dropna().unique():
            mask = age_bin == cat
            if mask.sum() >= args.min_count:
                _add_binary_feature(rows, y.loc[mask.index], mask, feature="age_bin", category=str(cat))

    # Sex
    if "sex" in df.columns:
        _add_categorical_feature(rows, y, df["sex"], "sex", args.min_count)

    # Race
    if "race" in df.columns:
        _add_categorical_feature(rows, y, df["race"], "race", args.min_count)
        race = df["race"].astype(str).str.strip().str.lower()
        mask_white = race == "white"
        if int(mask_white.sum()) >= args.min_count:
            _add_binary_feature(rows, y, mask_white, feature="race_white", category="white")

    # Ethnicity
    if "ethnicity" in df.columns:
        eth = df["ethnicity"].astype(str).str.strip().str.lower()
        hisp = eth.str.startswith("hispanic", na=False)
        if int(hisp.sum()) >= args.min_count:
            _add_binary_feature(rows, y, hisp, feature="ethnicity_hispanic", category="hispanic")
        _add_categorical_feature(rows, y, df["ethnicity"], "ethnicity", args.min_count)

    # Smoker
    if "tobacco_smoking_status_any" in df.columns:
        smoker = pd.to_numeric(df["tobacco_smoking_status_any"], errors="coerce")
        mask_smoker = smoker == 1
        if int(mask_smoker.sum()) >= args.min_count:
            _add_binary_feature(rows, y, mask_smoker, feature="smoker", category="yes")

    # Composite risk group
    if "age_years" in df.columns and "tobacco_smoking_status_any" in df.columns:
        age = pd.to_numeric(df["age_years"], errors="coerce")
        smoker = pd.to_numeric(df["tobacco_smoking_status_any"], errors="coerce")
        mask = (age >= float(args.risk_min_age)) & (smoker == 1)
        if args.risk_use_race and "race" in df.columns:
            race = df["race"].astype(str).str.strip().str.lower()
            mask = mask & (race == "white")
        if int(mask.sum()) >= args.min_count:
            _add_binary_feature(
                rows,
                y,
                mask,
                feature="risk_group",
                category="age>=min & smoker" + (" & white" if args.risk_use_race else ""),
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    # Print top groups by odds_ratio
    if not out_df.empty:
        top = out_df.sort_values("odds_ratio", ascending=False).head(10)
        print("Top groups by odds_ratio:")
        for _, row in top.iterrows():
            print(
                f"- {row['feature']}={row['category']}"
                f" n={int(row['n_total'])}"
                f" OR={row['odds_ratio']:.3f}"
                f" RR={row['risk_ratio']:.3f}"
            )
        print(f"OK Wrote {out_path}")


if __name__ == "__main__":
    main()
