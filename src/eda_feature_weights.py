from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _safe_or(pos_in: float, neg_in: float, pos_out: float, neg_out: float, smoothing: float = 0.5) -> float:
    return ((pos_in + smoothing) / (neg_in + smoothing)) / ((pos_out + smoothing) / (neg_out + smoothing))


def _logit(p: float) -> float:
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def _or_to_weight(or_value: float, factor: float) -> float:
    if not np.isfinite(or_value) or or_value <= 0:
        return 1.0
    log_or = float(np.log(or_value))
    return 1.0 + float(factor) * float(np.tanh(log_or))


def _missingness(df: pd.DataFrame, cols: List[str], domain_col: str) -> pd.DataFrame:
    rows = []
    for dom, sub in df.groupby(domain_col):
        for col in cols:
            if col not in sub.columns:
                continue
            s = sub[col]
            miss = pd.to_numeric(s, errors="coerce").isna().mean() if pd.api.types.is_numeric_dtype(s) else s.isna().mean()
            rows.append(
                {
                    "domain": str(dom),
                    "feature": col,
                    "missing_pct": float(miss),
                    "n": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def _numeric_bins(
    df: pd.DataFrame,
    feature: str,
    pos_mask: pd.Series,
    label_col: str,
    factor: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    series = pd.to_numeric(df[feature], errors="coerce")
    pos_vals = series[pos_mask].dropna()
    if pos_vals.empty:
        return pd.DataFrame(), {"feature": feature, "reason": "no_positive_values"}
    q1, q2, q3 = np.nanpercentile(pos_vals.to_numpy(), [25, 50, 75])
    bins = np.array([-np.inf, q1, q2, q3, np.inf], dtype=float)
    bins = np.unique(bins)
    if len(bins) < 3:
        return pd.DataFrame(), {"feature": feature, "reason": "insufficient_unique_bins"}
    labels = [f"Q{i + 1}" for i in range(len(bins) - 1)]
    binned = pd.cut(series, bins=bins, labels=labels, include_lowest=True)

    rows = []
    for label in labels:
        mask = binned == label
        pos_in = float((df.loc[mask, label_col] == 1).sum())
        neg_in = float((df.loc[mask, label_col] == 0).sum())
        pos_out = float((df[label_col] == 1).sum() - pos_in)
        neg_out = float((df[label_col] == 0).sum() - neg_in)
        total = pos_in + neg_in
        if total == 0:
            continue
        pos_rate = pos_in / total
        or_val = _safe_or(pos_in, neg_in, pos_out, neg_out)
        weight = _or_to_weight(or_val, factor)
        rows.append(
            {
                "feature": feature,
                "bin": str(label),
                "bin_low": float(bins[labels.index(label)]),
                "bin_high": float(bins[labels.index(label) + 1]),
                "n": int(total),
                "pos": int(pos_in),
                "neg": int(neg_in),
                "pos_rate": float(pos_rate),
                "odds_ratio": float(or_val),
                "weight": float(weight),
            }
        )
    return pd.DataFrame(rows), {"feature": feature, "bins": bins.tolist(), "labels": labels}


def _categorical_bins(
    df: pd.DataFrame,
    feature: str,
    label_col: str,
    factor: float,
    min_count: int,
    include_unknown: bool,
) -> pd.DataFrame:
    s = df[feature].astype(str).str.strip().str.lower()
    s = s.replace({"nan": np.nan, "none": np.nan, "": np.nan})
    if not include_unknown:
        s = s.replace({"unknown": np.nan})
    rows = []
    for cat, sub in df.assign(_v=s).groupby("_v"):
        if pd.isna(cat):
            continue
        pos_in = float((sub[label_col] == 1).sum())
        neg_in = float((sub[label_col] == 0).sum())
        total = pos_in + neg_in
        if total < min_count:
            continue
        pos_out = float((df[label_col] == 1).sum() - pos_in)
        neg_out = float((df[label_col] == 0).sum() - neg_in)
        pos_rate = pos_in / total if total else 0.0
        or_val = _safe_or(pos_in, neg_in, pos_out, neg_out)
        weight = _or_to_weight(or_val, factor)
        rows.append(
            {
                "feature": feature,
                "category": str(cat),
                "n": int(total),
                "pos": int(pos_in),
                "neg": int(neg_in),
                "pos_rate": float(pos_rate),
                "odds_ratio": float(or_val),
                "weight": float(weight),
            }
        )
    return pd.DataFrame(rows)


def _mode_targets(
    df: pd.DataFrame,
    numeric_bins: Dict[str, pd.DataFrame],
    numeric_meta: Dict[str, Dict[str, object]],
    cat_bins: Dict[str, pd.DataFrame],
    pos_mask: pd.Series,
    label_col: str,
    target_prob: float,
) -> Dict[str, object]:
    base_pos = float((df[label_col] == 1).mean())
    out: Dict[str, object] = {"base_pos_rate": base_pos, "target_prob": target_prob, "features": {}}

    for feature, bins_df in numeric_bins.items():
        if bins_df.empty:
            continue
        meta = numeric_meta.get(feature, {})
        bins = meta.get("bins")
        labels = meta.get("labels")
        if not bins or not labels:
            continue
        # mode in HCMI positives
        series = pd.to_numeric(df[feature], errors="coerce")
        binned = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
        pos_counts = binned[pos_mask].value_counts()
        if pos_counts.empty:
            continue
        mode_bin = str(pos_counts.idxmax())
        row = bins_df[bins_df["bin"] == mode_bin]
        if row.empty:
            continue
        or_val = float(row.iloc[0]["odds_ratio"])
        k = None
        reason = None
        if or_val > 1:
            k = (_logit(target_prob) - _logit(base_pos)) / float(np.log(or_val))
        else:
            reason = "mode_or<=1"
        out["features"][feature] = {
            "mode": mode_bin,
            "mode_pos_rate": float(row.iloc[0]["pos_rate"]),
            "mode_or": float(or_val),
            "scale_k": None if k is None else float(k),
            "reason": reason,
        }

    for feature, bins_df in cat_bins.items():
        if bins_df.empty:
            continue
        s = df[feature].astype(str).str.strip().str.lower()
        s = s.replace({"nan": np.nan, "none": np.nan, "": np.nan})
        pos_counts = s[pos_mask].value_counts()
        if pos_counts.empty:
            continue
        mode_cat = str(pos_counts.idxmax())
        row = bins_df[bins_df["category"] == mode_cat]
        if row.empty:
            continue
        or_val = float(row.iloc[0]["odds_ratio"])
        k = None
        reason = None
        if or_val > 1:
            k = (_logit(target_prob) - _logit(base_pos)) / float(np.log(or_val))
        else:
            reason = "mode_or<=1"
        out["features"][feature] = {
            "mode": mode_cat,
            "mode_pos_rate": float(row.iloc[0]["pos_rate"]),
            "mode_or": float(or_val),
            "scale_k": None if k is None else float(k),
            "reason": reason,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA para ponderadores estadisticos por feature.")
    ap.add_argument("--input", required=True, help="CSV combinado aligned.")
    ap.add_argument("--outdir", default="results/eda_weights", help="Carpeta salida.")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--domain-col", default="domain")
    ap.add_argument("--pos-domain", default="hcmi_tcga")
    ap.add_argument("--weight-factor", type=float, default=0.5)
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--include-unknown", action="store_true")
    ap.add_argument("--target-prob", type=float, default=0.70)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    numeric_cols = ["age_years", "height_last", "weight_last", "bmi_last", "pack_years_smoked_max"]
    categorical_cols = ["sex", "race", "ethnicity", "tobacco_smoking_status_any"]
    available = [c for c in numeric_cols + categorical_cols if c in df.columns]

    miss_df = _missingness(df, available, args.domain_col)
    miss_df.to_csv(outdir / "feature_missingness.csv", index=False)

    pos_mask = (df[args.label_col] == 1) & (df[args.domain_col].astype(str) == str(args.pos_domain))

    numeric_bins: Dict[str, pd.DataFrame] = {}
    numeric_meta: Dict[str, Dict[str, object]] = {}
    cat_bins: Dict[str, pd.DataFrame] = {}

    rows_num = []
    for feature in [c for c in numeric_cols if c in df.columns]:
        bins_df, meta = _numeric_bins(df, feature, pos_mask, args.label_col, args.weight_factor)
        if not bins_df.empty:
            rows_num.append(bins_df)
            numeric_bins[feature] = bins_df
        numeric_meta[feature] = meta
    if rows_num:
        pd.concat(rows_num, ignore_index=True).to_csv(outdir / "feature_numeric_bins.csv", index=False)

    rows_cat = []
    for feature in [c for c in categorical_cols if c in df.columns]:
        cat_df = _categorical_bins(
            df,
            feature,
            args.label_col,
            args.weight_factor,
            args.min_count,
            args.include_unknown,
        )
        if not cat_df.empty:
            rows_cat.append(cat_df)
            cat_bins[feature] = cat_df
    if rows_cat:
        pd.concat(rows_cat, ignore_index=True).to_csv(outdir / "feature_categorical_bins.csv", index=False)

    mode_targets = _mode_targets(
        df,
        numeric_bins,
        numeric_meta,
        cat_bins,
        pos_mask,
        args.label_col,
        args.target_prob,
    )
    with open(outdir / "feature_mode_targets.json", "w", encoding="utf-8") as f:
        json.dump(mode_targets, f, indent=2)

    print("OK EDA saved to", outdir)


if __name__ == "__main__":
    main()
