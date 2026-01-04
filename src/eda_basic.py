from __future__ import annotations
"""
EDA básico del dataset clínico (RAW) + gráfico tipo pirámide (Edad vs Sexo).

OBJETIVO:
- Validar cuántos pacientes reales se extrajeron desde el JSON (raw.csv)
- Revisar coherencia clínica básica:
    * Edad (en HCMI suele venir en días → se convierte a años)
    * Sexo
    * Distribución de label
- Generar un gráfico tipo pirámide:
    results/eda/age_sex_pyramid.png

USO:
python -m src.eda_basic

REQUISITOS:
- data/raw.csv debe existir
"""

from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt


DATA_PATH = Path("data/raw.csv")
OUTDIR = Path("results/eda")
OUT_PNG = OUTDIR / "age_sex_pyramid.png"


def _find_age_column(df: pd.DataFrame) -> str | None:
    # Preferimos explícitamente mean_age_at_dx
    for c in df.columns:
        if c.lower() == "mean_age_at_dx":
            return c
    # fallback: cualquier columna con "age" (si no existe la anterior)
    age_cols = [c for c in df.columns if "age" in c.lower()]
    return age_cols[0] if age_cols else None


def _find_sex_column(df: pd.DataFrame) -> str | None:
    # lista típica de columnas
    candidates = ["sex", "gender", "dem_gender", "demographic_gender", "patient_gender"]
    for name in candidates:
        for c in df.columns:
            if c.lower() == name:
                return c

    # fallback: cualquier columna que contenga sex o gender
    sex_cols = [c for c in df.columns if ("sex" in c.lower()) or ("gender" in c.lower())]
    return sex_cols[0] if sex_cols else None


def _normalize_sex(series: pd.Series) -> pd.Series:
    """
    Normaliza valores de sexo a: 'Male' / 'Female' / 'Unknown'
    """
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
            # valores raros los marcamos como Unknown para no romper
            out.append("Unknown")
    return pd.Series(out, index=series.index)


def plot_age_sex_pyramid(df: pd.DataFrame, age_years_col: str, sex_norm_col: str) -> None:
    """
    Crea una pirámide poblacional (edad vs sexo) y guarda un PNG.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Definir bins como el ejemplo (<20, 20-29, ..., 80+)
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 200]
    labels = ["Menos de 20", "Entre 20-29", "Entre 30-39", "Entre 40-49",
              "Entre 50-59", "Entre 60-69", "Entre 70-79", "80 y más"]

    df2 = df[[age_years_col, sex_norm_col]].dropna().copy()
    df2 = df2[df2[age_years_col].between(0, 120, inclusive="both")]  # filtro razonable

    df2["age_group"] = pd.cut(df2[age_years_col], bins=bins, labels=labels, right=False)

    # Conteos por sexo y grupo
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

    # Para pirámide: masculino negativo
    male_values = -male_counts.values
    female_values = female_counts.values
    y_pos = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, male_values)
    plt.barh(y_pos, female_values)

    plt.yticks(y_pos, labels)
    plt.axvline(0)

    # Ejes y títulos
    plt.xlabel("Recuento")
    plt.title("Distribución de pacientes según edad y sexo")
    plt.text(0.25, 1.02, "Masculino", transform=plt.gca().transAxes, ha="center")
    plt.text(0.75, 1.02, "Femenino", transform=plt.gca().transAxes, ha="center")

    # Ajustar ticks para mostrar valores positivos (aunque a la izquierda sean negativos)
    xlim = max(female_counts.max(), male_counts.max())
    plt.xlim(-xlim * 1.2, xlim * 1.2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("===== EDA BÁSICO =====")
    print(f"\nPacientes totales: {len(df)}")

    # -------------------------
    # Edad (días → años)
    # -------------------------
    age_col = _find_age_column(df)
    if age_col is None:
        print("\n⚠️ No se encontró columna de edad")
        return

    # Si es mean_age_at_dx, asumimos días (HCMI); si no, igual convertimos si parece grande
    age_raw = pd.to_numeric(df[age_col], errors="coerce")
    age_years = age_raw / 365.25  # conversión correcta

    df["age_years"] = age_years

    print("\n=== EDAD (CORREGIDA) ===")
    print(f"Columna usada (raw): {age_col}")
    print("⚠️ Interpretando como DÍAS y convirtiendo a AÑOS para el EDA.\n")
    print(f"Media (años):   {df['age_years'].mean():.2f}")
    print(f"Mediana (años): {df['age_years'].median():.2f}")
    print(f"Mín (años):     {df['age_years'].min():.1f}")
    print(f"Máx (años):     {df['age_years'].max():.1f}")

    # -------------------------
    # Sexo
    # -------------------------
    sex_col = _find_sex_column(df)
    if sex_col is None:
        print("\n⚠️ No se encontró columna de sexo; no se puede graficar la pirámide.")
        return

    df["sex_norm"] = _normalize_sex(df[sex_col])

    print("\n=== SEXO (NORMALIZADO) ===")
    print(f"Columna usada: {sex_col}")
    print(df["sex_norm"].value_counts(dropna=False))

    # -------------------------
    # Label
    # -------------------------
    if "label" in df.columns:
        print("\n=== LABEL ===")
        print(df["label"].value_counts(normalize=True))
    else:
        print("\n⚠️ No se encontró columna label")

    # -------------------------
    # Gráfico pirámide
    # -------------------------
    plot_age_sex_pyramid(df, age_years_col="age_years", sex_norm_col="sex_norm")
    print("\n📊 Pirámide guardada en:", OUT_PNG)

    print("\n=======================")


if __name__ == "__main__":
    main()