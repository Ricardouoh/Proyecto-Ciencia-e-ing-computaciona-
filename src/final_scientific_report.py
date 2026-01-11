from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

from src.preprocess import infer_columns, load_preprocessor, transform_with_loaded


def _load_schema_cols(data_dir: Path, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    schema_path = data_dir / "schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        numeric = [c for c in schema.get("numeric", []) if c in X.columns]
        categorical = [c for c in schema.get("categorical", []) if c in X.columns]
        if numeric or categorical:
            return numeric, categorical
    df_tmp = X.copy()
    df_tmp["label"] = 0
    return infer_columns(df_tmp, target="label")


def _prepare_matrix(
    pre,
    X: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    if hasattr(pre, "feature_names_in_"):
        required = list(pre.feature_names_in_)
        for col in required:
            if col not in X.columns:
                X[col] = np.nan
        X = X[required]
    return transform_with_loaded(pre, X, numeric_cols, categorical_cols)


def _score_metric(metric: str, y_true: np.ndarray, proba: np.ndarray) -> float:
    metric = metric.lower()
    if metric == "auprc":
        return float(average_precision_score(y_true, proba))
    if metric == "auroc":
        return float(roc_auc_score(y_true, proba))
    if metric == "brier":
        return float(np.mean((proba - y_true) ** 2))
    raise ValueError("metric debe ser auprc, auroc o brier")


def permutation_importance_raw(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    model,
    pre,
    numeric_cols: List[str],
    categorical_cols: List[str],
    metric: str = "auprc",
    n_repeats: int = 5,
    seed: int = 42,
) -> Tuple[float, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base_X = _prepare_matrix(pre, X_raw.copy(), numeric_cols, categorical_cols)
    base_proba = model.predict_proba(base_X)[:, 1]
    base_score = _score_metric(metric, y, base_proba)

    higher_is_better = metric != "brier"
    rows: List[Dict[str, float]] = []
    for col in X_raw.columns:
        deltas = []
        for _ in range(n_repeats):
            Xp = X_raw.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            Xt = _prepare_matrix(pre, Xp, numeric_cols, categorical_cols)
            proba = model.predict_proba(Xt)[:, 1]
            score = _score_metric(metric, y, proba)
            if higher_is_better:
                delta = base_score - score
            else:
                delta = score - base_score
            deltas.append(float(delta))
        rows.append(
            {
                "feature": col,
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas)),
            }
        )
    imp = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    return base_score, imp


def _age_bmi_report(
    df: pd.DataFrame,
    proba: np.ndarray,
    domain_col: str,
    unlabeled_domain: str,
) -> Tuple[pd.DataFrame, Optional[float], Optional[float]]:
    if domain_col not in df.columns:
        return pd.DataFrame(), None, None
    sub = df[df[domain_col].astype(str) == str(unlabeled_domain)].copy()
    if sub.empty:
        return pd.DataFrame(), None, None
    sub["proba"] = proba[sub.index]

    age = pd.to_numeric(sub.get("age_years"), errors="coerce")
    bmi = pd.to_numeric(sub.get("bmi_last"), errors="coerce")
    sub["age_years"] = age
    sub["bmi_last"] = bmi

    age_bins = [0, 40, 50, 60, 70, 80, np.inf]
    age_labels = ["<40", "40-49", "50-59", "60-69", "70-79", "80+"]
    sub["age_bin"] = pd.cut(age, bins=age_bins, labels=age_labels, right=False)

    bmi_bins = [0, 25, 30, np.inf]
    bmi_labels = ["Normal", "Sobrepeso", "Obeso"]
    sub["bmi_cat"] = pd.cut(bmi, bins=bmi_bins, labels=bmi_labels, right=False)

    grid = (
        sub.groupby(["age_bin", "bmi_cat"], observed=True)["proba"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "mean_proba"})
    )

    age_ge70 = sub[age >= 70]["proba"]
    age_lt40 = sub[age < 40]["proba"]
    mean_ge70 = float(age_ge70.mean()) if not age_ge70.empty else None
    mean_lt40 = float(age_lt40.mean()) if not age_lt40.empty else None
    return grid, mean_ge70, mean_lt40


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means.append(float(np.mean(sample)))
    mean = float(np.mean(values))
    low = float(np.percentile(means, 2.5))
    high = float(np.percentile(means, 97.5))
    return mean, low, high


def _age_decade_ci_report(
    df: pd.DataFrame,
    proba: np.ndarray,
    domain_col: str,
    unlabeled_domain: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if domain_col not in df.columns:
        return pd.DataFrame()
    sub = df[df[domain_col].astype(str) == str(unlabeled_domain)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["proba"] = proba[sub.index]
    age = pd.to_numeric(sub.get("age_years"), errors="coerce")
    sub["age_years"] = age

    bins = [20, 30, 40, 50, 60, 70, 80, 200]
    labels = ["20s", "30s", "40s", "50s", "60s", "70s", "80s+"]
    sub = sub[(age >= 20) & (age <= 120)]
    sub["age_decade"] = pd.cut(age, bins=bins, labels=labels, right=False)

    rows: List[Dict[str, float]] = []
    for label in labels:
        vals = sub.loc[sub["age_decade"] == label, "proba"].to_numpy()
        n = int(len(vals))
        mean, low, high = _bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "age_decade": label,
                "n": n,
                "mean_proba": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _plot_age_decade_ci(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    plot_df = df[np.isfinite(df["mean_proba"])].copy()
    if plot_df.empty:
        return
    x = np.arange(len(plot_df))
    y = plot_df["mean_proba"].to_numpy()
    yerr = np.vstack([y - plot_df["ci_low"].to_numpy(), plot_df["ci_high"].to_numpy() - y])
    plt.figure(figsize=(6.5, 4))
    plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
    plt.xticks(x, plot_df["age_decade"].tolist())
    plt.xlabel("Age decade (NHANES)")
    plt.ylabel("Mean predicted risk")
    plt.title("NHANES mean risk by age decade (95% CI)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _mode_or_none(series: Optional[pd.Series]) -> Optional[str]:
    if series is None:
        return None
    series = series.dropna().astype(str)
    if series.empty:
        return None
    counts = series.value_counts()
    return str(counts.index[0]) if not counts.empty else None


def _high_risk_profile(
    df: pd.DataFrame,
    proba: np.ndarray,
    domain_col: str,
    unlabeled_domain: str,
    quantile: float,
) -> Dict[str, Optional[float | str | int]]:
    if domain_col not in df.columns:
        return {}
    sub = df[df[domain_col].astype(str) == str(unlabeled_domain)].copy()
    if sub.empty:
        return {}
    sub["proba"] = proba[sub.index]
    threshold = float(sub["proba"].quantile(quantile))
    high = sub[sub["proba"] >= threshold].copy()
    if high.empty:
        return {}

    age = pd.to_numeric(high.get("age_years"), errors="coerce")
    bmi = pd.to_numeric(high.get("bmi_last"), errors="coerce")
    high["age_years"] = age
    high["bmi_last"] = bmi

    bmi_bins = [0, 25, 30, np.inf]
    bmi_labels = ["Normal", "Sobrepeso", "Obeso"]
    high["bmi_cat"] = pd.cut(bmi, bins=bmi_bins, labels=bmi_labels, right=False)

    return {
        "n": int(len(high)),
        "proba_threshold": threshold,
        "proba_mean": float(high["proba"].mean()),
        "proba_median": float(high["proba"].median()),
        "age_mean": float(age.mean()) if age.notna().any() else None,
        "age_median": float(age.median()) if age.notna().any() else None,
        "bmi_mean": float(bmi.mean()) if bmi.notna().any() else None,
        "bmi_median": float(bmi.median()) if bmi.notna().any() else None,
        "bmi_cat_mode": _mode_or_none(high["bmi_cat"]),
        "ethnicity_mode": _mode_or_none(high.get("ethnicity")),
        "race_mode": _mode_or_none(high.get("race")),
        "sex_mode": _mode_or_none(high.get("sex")),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Final PU scientific report.")
    ap.add_argument("--data-dir", default="data/processed_raw_full")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--metric", default="auprc", choices=["auprc", "auroc", "brier"])
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--age-ci-bootstrap", type=int, default=300)
    ap.add_argument("--high-risk-quantile", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unlabeled-domain", default="nhanes")
    ap.add_argument("--domain-col", default="domain")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    test_csv = data_dir / "aligned_test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"No se encontro {test_csv}")

    df = pd.read_csv(test_csv)
    if "label" not in df.columns:
        raise ValueError("No se encontro la columna label en aligned_test.csv")
    y = df["label"].astype(int).to_numpy()
    X_raw = df.drop(columns=["label"], errors="ignore")
    if args.domain_col in X_raw.columns:
        X_raw = X_raw.drop(columns=[args.domain_col])

    model = joblib.load(results_dir / "model.joblib")
    pre = load_preprocessor(results_dir / "preprocessor.joblib")
    numeric_cols, categorical_cols = _load_schema_cols(data_dir, X_raw)
    X_base = _prepare_matrix(pre, X_raw.copy(), numeric_cols, categorical_cols)
    proba = model.predict_proba(X_base)[:, 1]

    brier = float(np.mean((proba - y) ** 2))

    base_score, imp = permutation_importance_raw(
        X_raw,
        y,
        model,
        pre,
        numeric_cols,
        categorical_cols,
        metric=str(args.metric),
        n_repeats=int(args.n_repeats),
        seed=int(args.seed),
    )

    imp_path = results_dir / "eda" / "permutation_importance.csv"
    imp_path.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(imp_path, index=False)

    age_bmi_grid, mean_ge70, mean_lt40 = _age_bmi_report(
        df,
        proba,
        domain_col=str(args.domain_col),
        unlabeled_domain=str(args.unlabeled_domain),
    )
    grid_path = results_dir / "eda" / "nhanes_age_bmi_grid.csv"
    if not age_bmi_grid.empty:
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        age_bmi_grid.to_csv(grid_path, index=False)

    ratio = None
    if mean_ge70 is not None and mean_lt40 is not None and mean_lt40 > 0:
        ratio = float(mean_ge70 / mean_lt40)

    age_ci = _age_decade_ci_report(
        df,
        proba,
        domain_col=str(args.domain_col),
        unlabeled_domain=str(args.unlabeled_domain),
        n_boot=int(args.age_ci_bootstrap),
        seed=int(args.seed),
    )
    age_ci_path = results_dir / "eda" / "nhanes_age_decade_ci.csv"
    age_ci_plot = results_dir / "eda" / "nhanes_age_decade_ci.png"
    if not age_ci.empty:
        age_ci_path.parent.mkdir(parents=True, exist_ok=True)
        age_ci.to_csv(age_ci_path, index=False)
        _plot_age_decade_ci(age_ci, age_ci_plot)

    high_risk = _high_risk_profile(
        df,
        proba,
        domain_col=str(args.domain_col),
        unlabeled_domain=str(args.unlabeled_domain),
        quantile=float(args.high_risk_quantile),
    )

    top3 = imp.head(3).to_dict(orient="records")
    report = {
        "brier_score": brier,
        "importance_metric": str(args.metric),
        "importance_base_score": base_score,
        "top_features": top3,
        "risk_ratio_age_ge70_over_lt40": ratio,
        "age_ge70_mean_proba": mean_ge70,
        "age_lt40_mean_proba": mean_lt40,
        "permutation_importance_csv": str(imp_path),
        "nhanes_age_bmi_grid_csv": str(grid_path) if grid_path.exists() else None,
        "nhanes_age_decade_ci_csv": str(age_ci_path) if age_ci_path.exists() else None,
        "nhanes_age_decade_ci_png": str(age_ci_plot) if age_ci_plot.exists() else None,
        "nhanes_high_risk_profile": high_risk,
    }

    out_path = results_dir / "final_scientific_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Brier score:", brier)
    print("Age>=70 mean proba:", mean_ge70)
    print("Age<40 mean proba:", mean_lt40)
    print("Risk ratio (>=70/<40):", ratio)
    print("Top 3 features:", [r["feature"] for r in top3])
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
