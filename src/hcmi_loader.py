# src/hcmi_loader.py
from __future__ import annotations
"""
Convierte JSON clínico HCMI/CMDC (anidado) a CSVs tabulares normalizados.

Genera:
- cases.csv                      (1 fila por caso)
- exposures.csv                  (0..n por caso)
- family_histories.csv           (0..n por caso)
- diagnoses.csv                  (0..n por caso)
- treatments.csv                 (0..n por diagnóstico)
- pathology_details.csv          (0..n por diagnóstico)
- follow_ups.csv                 (0..n por caso)
- molecular_tests.csv            (0..n por follow_up)
- other_clinical_attributes.csv  (0..n por follow_up)

Uso:
  python -m src.hcmi_loader --in <archivo.json|carpeta> --outdir data/clinical
"""

from pathlib import Path
from typing import Any, Dict, List
import json
import pandas as pd


def _read_json_records(p: Path) -> List[Dict[str, Any]]:
    text = p.read_text(encoding="utf-8").strip()
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text = text + "]"
    text = text.replace("}\n{", "},\n{")
    return json.loads(text)


def _coerce(v):
    if isinstance(v, str):
        lv = v.lower().strip()
        if lv in ("true", "yes"):
            return 1
        if lv in ("false", "no"):
            return 0
    return v


def _only_scalars(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            continue
        out[prefix + k] = _coerce(v)
    return out


def _list(d: Dict[str, Any], key: str):
    v = d.get(key)
    return v if isinstance(v, list) else []


def flatten_hcmi(records: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    cases_rows: List[Dict[str, Any]] = []
    exposures_rows: List[Dict[str, Any]] = []
    fam_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    treat_rows: List[Dict[str, Any]] = []
    pathdet_rows: List[Dict[str, Any]] = []
    fu_rows: List[Dict[str, Any]] = []
    moltest_rows: List[Dict[str, Any]] = []
    other_attr_rows: List[Dict[str, Any]] = []

    for rec in records:
        case_submitter = rec.get("submitter_id")
        case_id = rec.get("case_id")

        base = _only_scalars(rec)
        base["case_submitter_id"] = case_submitter
        base["case_id"] = case_id
        if isinstance(rec.get("project"), dict):
            base["project_id"] = rec["project"].get("project_id")
        if isinstance(rec.get("demographic"), dict):
            dem = _only_scalars(rec["demographic"], prefix="dem_")
            base.update(dem)
        for k in ("disease_type", "primary_site", "index_date", "state"):
            base.setdefault(k, rec.get(k))
        cases_rows.append(base)

        for e in _list(rec, "exposures"):
            row = {"case_submitter_id": case_submitter, "case_id": case_id}
            row.update(_only_scalars(e))
            exposures_rows.append(row)

        for fh in _list(rec, "family_histories"):
            row = {"case_submitter_id": case_submitter, "case_id": case_id}
            row.update(_only_scalars(fh))
            fam_rows.append(row)

        for dg in _list(rec, "diagnoses"):
            dg_id = dg.get("diagnosis_id")
            drow = {"case_submitter_id": case_submitter, "case_id": case_id, "diagnosis_id": dg_id}
            drow.update(_only_scalars(dg))
            diag_rows.append(drow)

            for tr in _list(dg, "treatments"):
                trow = {"case_submitter_id": case_submitter, "case_id": case_id, "diagnosis_id": dg_id}
                trow.update(_only_scalars(tr))
                treat_rows.append(trow)

            for pdx in _list(dg, "pathology_details"):
                prow = {"case_submitter_id": case_submitter, "case_id": case_id, "diagnosis_id": dg_id}
                prow.update(_only_scalars(pdx))
                pathdet_rows.append(prow)

        for fu in _list(rec, "follow_ups"):
            fu_id = fu.get("follow_up_id")
            frow = {"case_submitter_id": case_submitter, "case_id": case_id, "follow_up_id": fu_id}
            frow.update(_only_scalars(fu))
            fu_rows.append(frow)

            for mt in _list(fu, "molecular_tests"):
                mrow = {"case_submitter_id": case_submitter, "case_id": case_id, "follow_up_id": fu_id}
                mrow.update(_only_scalars(mt))
                moltest_rows.append(mrow)

            for oa in _list(fu, "other_clinical_attributes"):
                orow = {"case_submitter_id": case_submitter, "case_id": case_id, "follow_up_id": fu_id}
                orow.update(_only_scalars(oa))
                other_attr_rows.append(orow)

    def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        key_order = [c for c in ("case_submitter_id", "case_id", "diagnosis_id", "follow_up_id") if c in df.columns]
        other_cols = [c for c in df.columns if c not in key_order]
        return df[key_order + other_cols]

    return {
        "cases": _df(cases_rows),
        "exposures": _df(exposures_rows),
        "family_histories": _df(fam_rows),
        "diagnoses": _df(diag_rows),
        "treatments": _df(treat_rows),
        "pathology_details": _df(pathdet_rows),
        "follow_ups": _df(fu_rows),
        "molecular_tests": _df(moltest_rows),
        "other_clinical_attributes": _df(other_attr_rows),
    }


def load_hcmi(source: str | Path) -> List[Dict[str, Any]]:
    src = Path(source)
    if src.is_dir():
        files = sorted(list(src.rglob("*.json"))) + sorted(list(src.rglob("*.jsonl"))) + sorted(list(src.rglob("*.ndjson")))
    else:
        files = [src]
    all_records: List[Dict[str, Any]] = []
    for p in files:
        all_records.extend(_read_json_records(p))
    return all_records


def _merge_with_existing(path: Path, new_df: pd.DataFrame, append: bool) -> pd.DataFrame:
    """Concatena con el CSV existente (si se solicita) y deduplica filas."""
    if not append:
        return new_df

    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if new_df is None or new_df.empty:
        return existing
    if existing is None or existing.empty:
        return new_df

    all_cols = list(dict.fromkeys(existing.columns.tolist() + [c for c in new_df.columns if c not in existing.columns]))
    existing_aligned = existing.reindex(columns=all_cols)
    new_aligned = new_df.reindex(columns=all_cols)

    combined = pd.concat([existing_aligned, new_aligned], ignore_index=True)
    return combined.drop_duplicates()


def save_tables(tables: Dict[str, pd.DataFrame], outdir: str | Path, *, append: bool = False) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = out / f"{name}.csv"
        merged = _merge_with_existing(path, df, append)
        if merged is None or merged.empty:
            path.write_text("", encoding="utf-8")
        else:
            merged.to_csv(path, index=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convierte JSON HCMI/CMDC a tablas CSV normalizadas.")
    ap.add_argument("--in", dest="inp", required=True, help="Archivo .json/.jsonl o carpeta con varios.")
    ap.add_argument("--outdir", required=True, help="Carpeta de salida (CSV por tabla).")
    ap.add_argument("--append", action="store_true", help="Anexar a los CSV existentes en vez de reemplazar.")
    args = ap.parse_args()

    records = load_hcmi(args.inp)
    tables = flatten_hcmi(records)
    save_tables(tables, args.outdir, append=args.append)

    print(f"Registros procesados: {len(records)}")
    for k, df in tables.items():
        n = 0 if df is None else len(df)
        print(f"  - {k}.csv: {n} filas")
