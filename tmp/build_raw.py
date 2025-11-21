import os, numpy as np, pandas as pd

# ----------------- utilidades -----------------
def with_case_key(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    if "case_id" in df.columns and df["case_id"].notna().any():
        df["case_key"] = df["case_id"].astype(str)
    elif "case_submitter_id" in df.columns:
        df["case_key"] = df["case_submitter_id"].astype(str)
    else:
        df["case_key"] = pd.RangeIndex(len(df)).astype(str)
    return df

def norm_text(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=str)
    return s.astype(str).str.lower().str.strip()

def safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

# ----------------- carga -----------------
C = r"data\clinical"
cases       = with_case_key(safe_read(os.path.join(C, "cases.csv")))
diagnoses   = with_case_key(safe_read(os.path.join(C, "diagnoses.csv")))
treatments  = with_case_key(safe_read(os.path.join(C, "treatments.csv")))
followups   = with_case_key(safe_read(os.path.join(C, "follow_ups.csv")))
exposures   = with_case_key(safe_read(os.path.join(C, "exposures.csv")))
famhist     = with_case_key(safe_read(os.path.join(C, "family_histories.csv")))
pathodet    = with_case_key(safe_read(os.path.join(C, "pathology_details.csv")))
oclattr     = with_case_key(safe_read(os.path.join(C, "other_clinical_attributes.csv")))
moltests    = with_case_key(safe_read(os.path.join(C, "molecular_tests.csv")))

if cases.empty:
    raise SystemExit("No se encontró data\\clinical\\cases.csv")

# ----------------- base: cases -----------------
df = cases.copy()

# etiqueta base con vital_status
if "dem_vital_status" in df.columns:
    vs = norm_text(df["dem_vital_status"])
    df["label"] = (vs != "alive").astype(int)
else:
    df["label"] = 0

# edad en años (si hay days_to_birth)
for col in ["age_years", "age_at_diagnosis", "days_to_birth"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
if "days_to_birth" in df.columns and "age_years" not in df.columns:
    df["age_years"] = (-df["days_to_birth"] / 365.25).round(1)

# normalizaciones sencillas
if "dem_gender" in df.columns: df["sex"] = norm_text(df["dem_gender"])
if "dem_ethnicity" in df.columns: df["ethnicity"] = norm_text(df["dem_ethnicity"])
if "disease_type" in df.columns: df["disease_type"] = norm_text(df["disease_type"])
if "primary_site" in df.columns: df["primary_site"] = norm_text(df["primary_site"])

# ----------------- exposures -----------------
if not exposures.empty:
    ex = exposures.copy()
    for c in ["pack_years_smoked", "tobacco_smoking_quit_year"]:
        if c in ex.columns: ex[c] = pd.to_numeric(ex[c], errors="coerce")
    ex["tobacco_smoking_status"] = norm_text(ex.get("tobacco_smoking_status", pd.Series(dtype=str)))
    agg_ex = (
        ex.groupby("case_key", dropna=False)
          .agg(
              pack_years_smoked_max=("pack_years_smoked", "max"),
              tobacco_smoking_status_any=("tobacco_smoking_status", lambda s: int(s.notna().any())),
          ).reset_index()
    )
    df = df.merge(agg_ex, how="left", on="case_key")

# ----------------- family histories -----------------
if not famhist.empty:
    fh = famhist.copy()
    fh["rel_hist_norm"] = norm_text(fh.get("relative_with_cancer_history", pd.Series(dtype=str)))
    fh["fh_pos"] = fh["rel_hist_norm"].isin(["1","yes","true","si","sí"]).astype(int)
    agg_fh = (
        fh.groupby("case_key", dropna=False)
          .agg(
              fam_hist_count=("family_history_id","count"),
              fam_hist_pos_any=("fh_pos","max"),
          ).reset_index()
    )
    df = df.merge(agg_fh, how="left", on="case_key")

# ----------------- diagnoses -----------------
if not diagnoses.empty:
    dx = diagnoses.copy()
    for ncol in ["age_at_diagnosis","days_to_diagnosis","days_to_last_follow_up"]:
        if ncol in dx.columns: dx[ncol] = pd.to_numeric(dx[ncol], errors="coerce")
    for c in ["ajcc_pathologic_stage","ajcc_clinical_stage","uicc_pathologic_stage",
              "uicc_clinical_stage","metastasis_at_diagnosis","prior_malignancy"]:
        if c in dx.columns: dx[c] = norm_text(dx[c])

    agg_dx = (
        dx.groupby("case_key", dropna=False)
          .agg(
              n_diagnoses=("diagnosis_id","count"),
              any_metastasis=("metastasis_at_diagnosis", lambda s: int(s.isin(["metastasis, nos","yes"]).any())),
              any_prior_malignancy=("prior_malignancy", lambda s: int(s.isin(["yes"]).any())),
              last_ajcc_stage=("ajcc_pathologic_stage", lambda s: s.dropna().iloc[-1] if s.dropna().size else pd.NA),
              mean_age_at_dx=("age_at_diagnosis","mean"),
          ).reset_index()
    )
    df = df.merge(agg_dx, how="left", on="case_key")

# ----------------- treatments -----------------
if not treatments.empty:
    tx = treatments.copy()
    tx["treatment_type_norm"] = norm_text(tx.get("treatment_type", pd.Series(dtype=str)))
    def has_word(s: pd.Series, word: str) -> int:
        return int(s.fillna("").str.contains(word, na=False).any())
    agg_tx = (
        tx.groupby("case_key", dropna=False)
          .agg(
              n_treatments=("treatment_id","count"),
              any_chemo=("treatment_type_norm", lambda s: has_word(s,"chemotherapy")),
              any_radiation=("treatment_type_norm", lambda s: has_word(s,"radiation")),
              any_surgery=("treatment_type_norm", lambda s: has_word(s,"surgery")),
              any_immuno=("treatment_type_norm", lambda s: has_word(s,"immunotherapy")),
              any_targeted=("treatment_type_norm", lambda s: has_word(s,"targeted")),
          ).reset_index()
    )
    df = df.merge(agg_tx, how="left", on="case_key")

# ----------------- follow-ups -----------------
if not followups.empty:
    fu = followups.copy()
    fu["progression_norm"] = norm_text(fu.get("progression_or_recurrence", pd.Series(dtype=str)))
    agg_fu = (
        fu.groupby("case_key", dropna=False)
          .agg(
              any_progression=("progression_norm", lambda s: int(s.isin(["yes"]).any())),
              last_days_to_follow_up=("days_to_follow_up", "max"),
          ).reset_index()
    )
    df = df.merge(agg_fu, how="left", on="case_key")
    df["label"] = np.where(df["any_progression"].fillna(0).astype(int)==1, 1, df["label"].astype(int))

# ----------------- pathology details -----------------
if not pathodet.empty:
    pdx = pathodet.copy()
    for c in ["vascular_invasion_present","lymphatic_invasion_present","perineural_invasion_present"]:
        if c in pdx.columns: pdx[c] = norm_text(pdx[c])
    agg_pd = (
        pdx.groupby("case_key", dropna=False)
           .agg(
               vascular_invasion_any=("vascular_invasion_present", lambda s: int(s.isin(["yes"]).any())),
               lymphatic_invasion_any=("lymphatic_invasion_present", lambda s: int(s.isin(["yes"]).any())),
               perineural_invasion_any=("perineural_invasion_present", lambda s: int(s.isin(["yes"]).any())),
           ).reset_index()
    )
    df = df.merge(agg_pd, how="left", on="case_key")

# ----------------- other clinical attributes -----------------
if not oclattr.empty:
    oa = oclattr.copy()
    for c in ["height","weight","bmi"]:
        if c in oa.columns: oa[c] = pd.to_numeric(oa[c], errors="coerce")
    agg_oa = (
        oa.groupby("case_key", dropna=False)
          .agg(
              height_last=("height","last"),
              weight_last=("weight","last"),
              bmi_last=("bmi","last"),
          ).reset_index()
    )
    df = df.merge(agg_oa, how="left", on="case_key")

# ----------------- molecular tests -----------------
if not moltests.empty:
    mt = moltests.copy()
    agg_mt = (
        mt.groupby("case_key", dropna=False)
          .agg(n_molecular_tests=("molecular_test_id","count"))
          .reset_index()
    )
    df = df.merge(agg_mt, how="left", on="case_key")

# ----------------- guardar -----------------
out = r"data\raw.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)

print("✔ Tabla plana guardada en:", out)
print("   Filas:", len(df), " Columnas:", len(df.columns))
print("   Tiene 'label':", "label" in df.columns)
