from __future__ import annotations
"""
Basic EDA for clinical datasets with an age/sex pyramid plot.

Defaults:
- input: data/training/raw.csv
- output: results/eda/age_sex_pyramid.png

Use --only-cancer to filter label == 1.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DATA_PATH = Path("data/training/raw.csv")
DEFAULT_OUTDIR = Path("results/eda")
DEFAULT_OUT_PNG = DEFAULT_OUTDIR / "age_sex_pyramid.png"


def _find_age_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() == "mean_age_at_dx":
            return c
    age_cols = [c for c in df.columns if "age" in c.lower()]
    return age_cols[0] if age_cols else None


def _find_sex_column(df: pd.DataFrame) -> str | None:
    candidates = ["sex", "gender", "dem_gender", "demographic_gender", "patient_gender"]
    for name in candidates:
        for c in df.columns:
            if c.lower() == name:
                return c
    sex_cols = [c for c in df.columns if ("sex" in c.lower()) or ("gender" in c.lower())]
    return sex_cols[0] if sex_cols else None


def _normalize_sex(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    male_set = {"m", "male", "masculino", "man"}
    female_set = {"f", "female", "femenino", "woman"}
    out = []
    for v in s.tolist():
        if v in male_set:
            out.append("Male")
        elif v in female_set:
            out.append("Female")
        elif v in ("nan", "none", "", "unknown", "not reported", "na"):
            out.append("Unknown")
        else:
            out.append("Unknown")
    return pd.Series(out, index=series.index)


def plot_age_sex_pyramid(
    df: pd.DataFrame,
    age_years_col: str,
    sex_norm_col: str,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    bins = [0, 20, 30, 40, 50, 60, 70, 80, 200]
    labels = [
        "Menos de 20",
        "Entre 20-29",
        "Entre 30-39",
        "Entre 40-49",
        "Entre 50-59",
        "Entre 60-69",
        "Entre 70-79",
        "80 y mas",
    ]

    df2 = df[[age_years_col, sex_norm_col]].dropna().copy()
    df2 = df2[df2[age_years_col].between(0, 120, inclusive="both")]
    df2["age_group"] = pd.cut(df2[age_years_col], bins=bins, labels=labels, right=False)

    male_counts = (
        df2[df2[sex_norm_col] == "Male"]["age_group"]
        .value_counts()
        .reindex(labels, fill_value=0)
    )
    female_counts = (
        df2[df2[sex_norm_col] == "Female"]["age_group"]
        .value_counts()
        .reindex(labels, fill_value=0)
    )

    male_values = -male_counts.values
    female_values = female_counts.values
    y_pos = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, male_values)
    plt.barh(y_pos, female_values)

    plt.yticks(y_pos, labels)
    plt.axvline(0)

    plt.xlabel("Recuento")
    plt.title("Distribucion de pacientes segun edad y sexo")
    plt.text(0.25, 1.02, "Masculino", transform=plt.gca().transAxes, ha="center")
    plt.text(0.75, 1.02, "Femenino", transform=plt.gca().transAxes, ha="center")

    xlim = max(female_counts.max(), male_counts.max())
    plt.xlim(-xlim * 1.2, xlim * 1.2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="EDA basico: piramide edad vs sexo.")
    ap.add_argument("--csv", default=str(DEFAULT_DATA_PATH), help="CSV de entrada.")
    ap.add_argument("--out", default=str(DEFAULT_OUT_PNG), help="PNG de salida.")
    ap.add_argument("--only-cancer", action="store_true", help="Filtra solo label == 1.")
    ap.add_argument("--only-noncancer", action="store_true", help="Filtra solo label == 0.")
    ap.add_argument("--label-value", type=int, default=None, help="Filtra por valor de label (0/1).")
    ap.add_argument("--label-col", default="label", help="Columna label para filtrar.")
    args = ap.parse_args()

    data_path = Path(args.csv)
    out_png = Path(args.out)

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontro {data_path}")

    df = pd.read_csv(data_path)

    label_value = args.label_value
    if label_value is None:
        if args.only_cancer and args.only_noncancer:
            raise ValueError("No se puede usar --only-cancer y --only-noncancer al mismo tiempo")
        if args.only_cancer:
            label_value = 1
        elif args.only_noncancer:
            label_value = 0

    if label_value is not None:
        if args.label_col not in df.columns:
            raise ValueError(f"No se encontro la columna label '{args.label_col}' para filtrar")
        df = df[df[args.label_col] == label_value].copy()

    print("===== EDA BASICO =====")
    print(f"\nPacientes totales: {len(df)}")

    age_col = _find_age_column(df)
    if age_col is None:
        print("\nNo se encontro columna de edad")
        return

    age_raw = pd.to_numeric(df[age_col], errors="coerce")
    median_age = age_raw.dropna().median()
    treat_as_days = str(age_col).lower() == "mean_age_at_dx" or (median_age is not None and median_age > 200)
    age_years = age_raw / 365.25 if treat_as_days else age_raw

    df["age_years"] = age_years

    print("\n=== EDAD ===")
    print(f"Columna usada (raw): {age_col}")
    if treat_as_days:
        print("Interpretando como DIAS y convirtiendo a ANOS para el EDA.\n")
    else:
        print("Interpretando como ANOS para el EDA.\n")
    print(f"Media (anos):   {df['age_years'].mean():.2f}")
    print(f"Mediana (anos): {df['age_years'].median():.2f}")
    print(f"Min (anos):     {df['age_years'].min():.1f}")
    print(f"Max (anos):     {df['age_years'].max():.1f}")

    sex_col = _find_sex_column(df)
    if sex_col is None:
        print("\nNo se encontro columna de sexo; no se puede graficar la piramide.")
        return

    df["sex_norm"] = _normalize_sex(df[sex_col])

    print("\n=== SEXO (NORMALIZADO) ===")
    print(f"Columna usada: {sex_col}")
    print(df["sex_norm"].value_counts(dropna=False))

    if args.label_col in df.columns:
        print("\n=== LABEL ===")
        print(df[args.label_col].value_counts(normalize=True))

    plot_age_sex_pyramid(df, age_years_col="age_years", sex_norm_col="sex_norm", out_png=out_png)
    print("\nPiramide guardada en:", out_png)
    print("\n=======================")


if __name__ == "__main__":
    main()
