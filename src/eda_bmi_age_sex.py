from __future__ import annotations
"""
Generate BMI vs age plots by sex, split by cancer label.

Default input: data/processed_nosmoke/combined_aligned.csv
Outputs:
  results/eda/bmi_age_sex_by_label.png
  results/eda/bmi_age_sex_trend.png
"""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _normalize_sex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sex" not in df.columns and "gender" in df.columns:
        df["sex"] = df["gender"]
    return df


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_plots(
    csv_path: Path,
    outdir: Path,
    *,
    domain: str | None = None,
    age_bin: int = 5,
) -> None:
    df = pd.read_csv(csv_path)
    df = _normalize_sex(df)

    _require_columns(df, ["age_years", "bmi_last", "sex", "label"])

    if domain and "domain" in df.columns:
        df = df[df["domain"].astype(str) == str(domain)].copy()

    df = df.dropna(subset=["age_years", "bmi_last", "sex", "label"]).copy()
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    df["bmi_last"] = pd.to_numeric(df["bmi_last"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    df = df.dropna(subset=["age_years", "bmi_last", "label"]).copy()

    df["label_name"] = df["label"].map({0: "No cancer", 1: "Cancer"}).fillna(df["label"].astype(str))

    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Scatter: BMI vs age, color by sex, panel by label
    g = sns.relplot(
        data=df,
        x="age_years",
        y="bmi_last",
        hue="sex",
        col="label_name",
        kind="scatter",
        alpha=0.35,
        s=18,
        height=4.5,
        aspect=1.1,
    )
    g.set_axis_labels("Age (years)", "BMI")
    g.set_titles("{col_name}")
    g.fig.suptitle("BMI vs age by sex", y=1.02)
    g.fig.tight_layout()
    g.savefig(outdir / "bmi_age_sex_by_label.png", dpi=150)

    # Trend: mean BMI by age bins and sex, split by label
    df["age_bin"] = (df["age_years"] // int(age_bin)) * int(age_bin)
    agg = (
        df.groupby(["label_name", "sex", "age_bin"], as_index=False)["bmi_last"]
        .mean()
        .rename(columns={"bmi_last": "bmi_mean"})
    )

    plt.figure(figsize=(8, 4.8))
    sns.lineplot(
        data=agg,
        x="age_bin",
        y="bmi_mean",
        hue="sex",
        style="label_name",
        markers=True,
    )
    plt.xlabel(f"Age (years) - bin {age_bin}")
    plt.ylabel("BMI mean")
    plt.title("BMI mean by age, sex, and label")
    plt.tight_layout()
    plt.savefig(outdir / "bmi_age_sex_trend.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot BMI vs age by sex for cancer vs non-cancer.")
    ap.add_argument(
        "--input",
        default="data/processed_nosmoke/combined_aligned.csv",
        help="Input CSV path.",
    )
    ap.add_argument(
        "--outdir",
        default="results/eda",
        help="Output folder for PNG files.",
    )
    ap.add_argument(
        "--domain",
        default=None,
        help="Optional domain filter (e.g. hcmi_tcga).",
    )
    ap.add_argument(
        "--age-bin",
        type=int,
        default=5,
        help="Age bin size for trend plot (default 5).",
    )
    args = ap.parse_args()

    make_plots(
        csv_path=Path(args.input),
        outdir=Path(args.outdir),
        domain=args.domain,
        age_bin=int(args.age_bin),
    )

    print(f"OK -> {Path(args.outdir) / 'bmi_age_sex_by_label.png'}")
    print(f"OK -> {Path(args.outdir) / 'bmi_age_sex_trend.png'}")


if __name__ == "__main__":
    main()
