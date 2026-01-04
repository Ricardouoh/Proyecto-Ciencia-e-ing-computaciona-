from __future__ import annotations
"""
EDA and classification analysis for data/raw.csv.

Outputs (default):
  results/analysis/
    summary.json
    missingness.csv
    missingness_top20.png
    <prefix>_label_distribution.png
    <prefix>_label_rate_by_<group>.png
    <prefix>_label_box_<feature>.png
    <prefix>_label_hist_<feature>.png
    <prefix>_corr_heatmap.png
    <prefix>_roc_curve.png
    <prefix>_pr_curve.png
    <prefix>_confusion_matrix.png
    <prefix>_classification_metrics.json
    <prefix>_classification_metrics.csv
    <prefix>_feature_importance.csv

Optional regression outputs remain available if --mode regression is used.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_LIKE_COLUMNS = {
    "case_id",
    "case_submitter_id",
    "submitter_id",
    "dem_demographic_id",
    "dem_submitter_id",
    "case_key",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _drop_id_like(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cols = [c for c in df.columns if c in ID_LIKE_COLUMNS or c.endswith("_id")]
    if not cols:
        return df, []
    return df.drop(columns=cols, errors="ignore"), cols


def _drop_constant_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    to_drop = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if not to_drop:
        return df, []
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def _drop_high_missing(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    miss = df.isna().mean()
    to_drop = miss[miss > threshold].index.tolist()
    if not to_drop:
        return df, []
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def _coerce_numeric_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
            continue
        if df[c].dtype == object:
            ser = pd.to_numeric(df[c], errors="coerce")
            non_empty = df[c].notna().sum()
            ratio = ser.notna().sum() / non_empty if non_empty else 0.0
            if ratio >= 0.9:
                df[c] = ser
                num_cols.append(c)
            else:
                cat_cols.append(c)
        else:
            cat_cols.append(c)
    return df, num_cols, cat_cols


def _drop_high_cardinality(
    df: pd.DataFrame,
    cat_cols: List[str],
    max_unique: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    to_drop = []
    for c in cat_cols:
        try:
            nunique = df[c].nunique(dropna=True)
        except TypeError:
            continue
        if nunique > max_unique:
            to_drop.append(c)
    if not to_drop:
        return df, cat_cols, []
    df = df.drop(columns=to_drop, errors="ignore")
    cat_cols = [c for c in cat_cols if c not in to_drop]
    return df, cat_cols, to_drop


def _make_ohe() -> OneHotEncoder:
    for kwargs in (
        dict(handle_unknown="ignore", sparse_output=False, min_frequency=0.01),
        dict(handle_unknown="ignore", sparse=False, min_frequency=0.01),
        dict(handle_unknown="ignore", sparse_output=False),
        dict(handle_unknown="ignore", sparse=False),
    ):
        try:
            return OneHotEncoder(**kwargs)
        except TypeError:
            continue
    return OneHotEncoder(handle_unknown="ignore")


def _make_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
) -> ColumnTransformer:
    transformers = []
    if num_cols:
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", _make_ohe()),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _get_feature_names(
    preprocessor: ColumnTransformer,
    num_cols: List[str],
    cat_cols: List[str],
) -> List[str]:
    names: List[str] = []
    if "num" in preprocessor.named_transformers_:
        names.extend(num_cols)
    if "cat" in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        try:
            cat_names = ohe.get_feature_names_out(cat_cols)
        except AttributeError:
            cat_names = ohe.get_feature_names(cat_cols)
        names.extend(list(cat_names))
    return names


def _save_missingness(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    miss_df = pd.DataFrame({
        "column": miss.index,
        "missing_rate": miss.values,
    })
    miss_df.to_csv(outdir / "missingness.csv", index=False)

    top = miss_df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x="missing_rate", y="column", data=top, color="#4c72b0")
    plt.xlabel("missing_rate")
    plt.ylabel("column")
    plt.title("Top 20 missingness")
    plt.tight_layout()
    plt.savefig(outdir / "missingness_top20.png", dpi=150)
    plt.close()
    return miss_df


def _prefix_name(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _norm_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def _filter_colon_cases(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    info = {"filtered": 0, "input_rows": int(len(df))}
    mask = pd.Series(False, index=df.index)
    for col in ("disease_type", "primary_site"):
        if col in df.columns:
            text = _norm_text(df[col])
            mask = mask | text.str.contains(r"colon|colorect", regex=True, na=False)
    if mask.any():
        df = df[mask].copy()
        info["filtered"] = 1
    info["output_rows"] = int(len(df))
    return df, info


def _convert_age_days_to_years(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    dropped: List[str] = []

    if "dem_days_to_birth" in df.columns:
        days = pd.to_numeric(df["dem_days_to_birth"], errors="coerce")
        df["age_years"] = (-days / 365.25).round(1)
        dropped.append("dem_days_to_birth")

    if "mean_age_at_dx" in df.columns:
        days = pd.to_numeric(df["mean_age_at_dx"], errors="coerce")
        df["mean_age_at_dx_years"] = (days / 365.25).round(1)
        dropped.append("mean_age_at_dx")

    return df, dropped


def _plot_target_hist(df: pd.DataFrame, target: str, outdir: Path) -> None:
    data = df[target].dropna()
    if data.empty:
        return
    plt.figure(figsize=(7, 4))
    sns.histplot(data, bins=30, kde=True, color="#55a868")
    plt.title(f"{target} distribution")
    plt.xlabel(target)
    plt.tight_layout()
    plt.savefig(outdir / f"target_{target}_hist.png", dpi=150)
    plt.close()


def _plot_label_distribution(
    df: pd.DataFrame,
    label_col: str,
    outdir: Path,
    prefix: str = "",
) -> None:
    if label_col not in df.columns:
        return
    counts = df[label_col].value_counts(dropna=False).sort_index()
    if counts.empty:
        return
    plt.figure(figsize=(4.5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#4c72b0")
    plt.xlabel(label_col)
    plt.ylabel("count")
    plt.title("Label distribution")
    plt.tight_layout()
    plt.savefig(outdir / _prefix_name(prefix, "label_distribution.png"), dpi=150)
    plt.close()


def _plot_box_by_label(
    df: pd.DataFrame,
    target: str,
    outdir: Path,
    prefix: str = "",
) -> None:
    if "label" not in df.columns:
        return
    data = df[[target, "label"]].dropna()
    if data.empty:
        return
    plt.figure(figsize=(5.5, 4))
    ax = sns.boxplot(x="label", y=target, data=data, hue="label", palette="Set2")
    if ax.legend_ is not None:
        ax.legend_.remove()
    plt.title(f"{target} by label")
    plt.tight_layout()
    plt.savefig(outdir / _prefix_name(prefix, f"label_box_{target}.png"), dpi=150)
    plt.close()


def _plot_hist_by_label(
    df: pd.DataFrame,
    target: str,
    outdir: Path,
    prefix: str = "",
) -> None:
    if "label" not in df.columns:
        return
    data = df[[target, "label"]].dropna()
    if data.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=data,
        x=target,
        hue="label",
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        palette="Set2",
    )
    plt.title(f"Distribucion de {target} por label")
    plt.tight_layout()
    plt.savefig(outdir / _prefix_name(prefix, f"label_hist_{target}.png"), dpi=150)
    plt.close()


def _plot_group_means(
    df: pd.DataFrame,
    target: str,
    group_col: str,
    outdir: Path,
    top_n: int = 10,
) -> None:
    if group_col not in df.columns:
        return
    data = df[[target, group_col]].dropna()
    if data.empty:
        return
    counts = data[group_col].value_counts()
    top_groups = counts.head(top_n).index
    data = data[data[group_col].isin(top_groups)]
    if data[group_col].nunique() < 2:
        return
    means = data.groupby(group_col)[target].mean().sort_values()
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=means.index, y=means.values, color="#c44e52")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(f"mean {target}")
    plt.title(f"{target} by {group_col} (top {top_n})")
    plt.tight_layout()
    plt.savefig(outdir / f"target_{target}_by_{group_col}.png", dpi=150)
    plt.close()


def _plot_label_rate_by_group(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    outdir: Path,
    prefix: str = "",
    top_n: int = 10,
) -> None:
    if group_col not in df.columns or label_col not in df.columns:
        return
    data = df[[label_col, group_col]].dropna()
    if data.empty:
        return
    counts = data[group_col].value_counts()
    top_groups = counts.head(top_n).index
    data = data[data[group_col].isin(top_groups)]
    if data[group_col].nunique() < 2:
        return
    rates = data.groupby(group_col)[label_col].mean().sort_values(ascending=False)
    group_labels = {
        "sex": "Sexo",
        "disease_type": "Tipo de enfermedad",
        "primary_site": "Sitio primario",
        "ethnicity": "Etnicidad",
    }
    group_title = group_labels.get(group_col, group_col)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=rates.index, y=rates.values, color="#55a868")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(f"Tasa {label_col}=1")
    plt.xlabel(group_title)
    plt.title(f"Tasa de cancer por {group_title} (top {top_n}, ordenado)")
    plt.tight_layout()
    plt.savefig(outdir / _prefix_name(prefix, f"label_rate_by_{group_col}.png"), dpi=150)
    plt.close()


def _plot_top_correlations(
    df: pd.DataFrame,
    target: str,
    outdir: Path,
    top_k: int = 4,
) -> None:
    tmp = df[[c for c in df.columns if c != target] + [target]].copy()
    tmp, num_cols, _ = _coerce_numeric_columns(tmp)
    if target not in num_cols:
        num_cols.append(target)
    num_cols = [c for c in num_cols if c in tmp.columns]
    if len(num_cols) < 3:
        return
    corr = tmp[num_cols].corr()
    if target not in corr.columns:
        return
    top = corr[target].drop(target).abs().sort_values(ascending=False).head(top_k)
    top_features = list(top.index)

    # heatmap
    subset = [target] + top_features
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr.loc[subset, subset],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"{target} correlation heatmap")
    plt.tight_layout()
    plt.savefig(outdir / f"target_{target}_corr_heatmap.png", dpi=150)
    plt.close()

    # scatter plots
    for feat in top_features:
        data = tmp[[target, feat]].dropna()
        if data.empty:
            continue
        plt.figure(figsize=(5.5, 4))
        sns.regplot(x=feat, y=target, data=data, scatter_kws={"s": 18, "alpha": 0.7})
        plt.title(f"{target} vs {feat}")
        plt.tight_layout()
        plt.savefig(outdir / f"target_{target}_scatter_{feat}.png", dpi=150)
    plt.close()


def _prepare_features(
    df: pd.DataFrame,
    target: str,
    max_missing: float,
    max_unique_cat: int,
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, List[str]]]:
    X = df.drop(columns=[target])
    info: Dict[str, List[str]] = {}

    X, dropped_id = _drop_id_like(X)
    info["dropped_id_like"] = dropped_id

    X, dropped_missing = _drop_high_missing(X, max_missing)
    info["dropped_high_missing"] = dropped_missing

    X, dropped_const = _drop_constant_cols(X)
    info["dropped_constant"] = dropped_const

    X, num_cols, cat_cols = _coerce_numeric_columns(X)
    X, cat_cols, dropped_card = _drop_high_cardinality(X, cat_cols, max_unique_cat)
    info["dropped_high_cardinality"] = dropped_card

    return X, num_cols, cat_cols, info


def run_regression(
    df: pd.DataFrame,
    target: str,
    outdir: Path,
    test_size: float,
    seed: int,
    max_missing: float,
    max_unique_cat: int,
) -> Dict[str, float]:
    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].notna()].copy()
    if len(df) < 30:
        return {"target": target, "n": len(df), "status": "skipped_low_rows"}

    X, num_cols, cat_cols, info = _prepare_features(
        df,
        target,
        max_missing=max_missing,
        max_unique_cat=max_unique_cat,
    )
    if not num_cols and not cat_cols:
        return {"target": target, "n": len(df), "status": "skipped_no_features"}

    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    pre = _make_preprocessor(num_cols, cat_cols)
    model = Ridge(alpha=1.0, random_state=seed)
    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)

    metrics = {
        "target": target,
        "n": int(len(df)),
        "r2_test": float(r2_score(y_test, y_pred)),
        "mae_test": float(mean_absolute_error(y_test, y_pred)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
        "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "n_features": int(pipe.named_steps["model"].coef_.shape[0]),
        "dropped_id_like": len(info["dropped_id_like"]),
        "dropped_high_missing": len(info["dropped_high_missing"]),
        "dropped_constant": len(info["dropped_constant"]),
        "dropped_high_cardinality": len(info["dropped_high_cardinality"]),
    }

    # predictions vs actual
    plt.figure(figsize=(5.5, 4))
    plt.scatter(y_test, y_pred, s=18, alpha=0.7)
    plt.xlabel("actual")
    plt.ylabel("predicted")
    plt.title(f"{target} - predicted vs actual")
    plt.tight_layout()
    plt.savefig(outdir / f"target_{target}_pred_vs_actual.png", dpi=150)
    plt.close()

    # coefficients
    pre = pipe.named_steps["preprocess"]
    names = _get_feature_names(pre, num_cols, cat_cols)
    coefs = pipe.named_steps["model"].coef_
    coef_df = pd.DataFrame({
        "feature": names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)
    coef_df.to_csv(outdir / f"coefficients_{target}.csv", index=False)

    return metrics


def _plot_corr_heatmap(
    df: pd.DataFrame,
    outdir: Path,
    prefix: str = "",
    top_k: int = 12,
) -> None:
    tmp, num_cols, _ = _coerce_numeric_columns(df)
    if len(num_cols) < 3:
        return
    corr = tmp[num_cols].corr()
    if corr.empty:
        return
    # pick top_k most variable columns by variance
    variances = tmp[num_cols].var(numeric_only=True).sort_values(ascending=False)
    cols = list(variances.head(top_k).index)
    corr = corr.loc[cols, cols]
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlacion (variables numericas principales)")
    plt.tight_layout()
    plt.savefig(outdir / _prefix_name(prefix, "corr_heatmap.png"), dpi=150)
    plt.close()


def _plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    outdir: Path,
    prefix: str = "",
) -> None:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        return
    plt.figure(figsize=(5.5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / _prefix_name(prefix, "roc_curve.png"), dpi=150)
    plt.close()


def _plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    outdir: Path,
    prefix: str = "",
) -> None:
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auprc = average_precision_score(y_true, y_proba)
    except Exception:
        return
    plt.figure(figsize=(5.5, 4))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outdir / _prefix_name(prefix, "pr_curve.png"), dpi=150)
    plt.close()


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outdir: Path,
    prefix: str = "",
) -> None:
    try:
        cm = confusion_matrix(y_true, y_pred)
    except Exception:
        return
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(outdir / _prefix_name(prefix, "confusion_matrix.png"), dpi=150)
    plt.close()


def run_classification(
    df: pd.DataFrame,
    label_col: str,
    outdir: Path,
    prefix: str,
    test_size: float,
    seed: int,
    max_missing: float,
    max_unique_cat: int,
) -> Dict[str, float]:
    if label_col not in df.columns:
        return {"status": "label_missing"}

    df = df.copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df[df[label_col].notna()].copy()
    if len(df) < 30:
        return {"status": "skipped_low_rows", "n": int(len(df))}

    y = df[label_col].astype(int)
    if y.nunique() < 2:
        return {"status": "single_class", "n": int(len(df))}

    X, num_cols, cat_cols, info = _prepare_features(
        df,
        label_col,
        max_missing=max_missing,
        max_unique_cat=max_unique_cat,
    )
    if not num_cols and not cat_cols:
        return {"status": "skipped_no_features", "n": int(len(df))}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    pre = _make_preprocessor(num_cols, cat_cols)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,
    )
    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "n": int(len(df)),
        "auroc": float(roc_auc_score(y_test, y_proba)),
        "auprc": float(average_precision_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_features": int(pipe.named_steps["model"].coef_.shape[1]),
        "dropped_id_like": len(info["dropped_id_like"]),
        "dropped_high_missing": len(info["dropped_high_missing"]),
        "dropped_constant": len(info["dropped_constant"]),
        "dropped_high_cardinality": len(info["dropped_high_cardinality"]),
    }

    _plot_roc_curve(y_test.values, y_proba, outdir, prefix=prefix)
    _plot_pr_curve(y_test.values, y_proba, outdir, prefix=prefix)
    _plot_confusion_matrix(y_test.values, y_pred, outdir, prefix=prefix)

    pre = pipe.named_steps["preprocess"]
    names = _get_feature_names(pre, num_cols, cat_cols)
    coefs = pipe.named_steps["model"].coef_.ravel()
    coef_df = pd.DataFrame({
        "feature": names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)
    coef_df.to_csv(outdir / _prefix_name(prefix, "feature_importance.csv"), index=False)

    return metrics


def _write_summary(df: pd.DataFrame, miss_df: pd.DataFrame, outdir: Path) -> None:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    summary = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "num_cols": len(num_cols),
        "cat_cols": len(cat_cols),
        "missing_top10": miss_df.head(10).to_dict(orient="records"),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _select_targets(
    df: pd.DataFrame,
    main_target: str,
    extra_targets: List[str],
    max_missing: float,
    min_rows: int,
) -> List[str]:
    targets = []
    for t in [main_target] + extra_targets:
        if t not in df.columns:
            continue
        miss_rate = df[t].isna().mean()
        if miss_rate > max_missing:
            continue
        if df[t].notna().sum() < min_rows:
            continue
        targets.append(t)
    # de-duplicate
    out = []
    for t in targets:
        if t not in out:
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA and classification on raw CSV.")
    ap.add_argument("--csv", default="data/raw.csv", help="Input CSV.")
    ap.add_argument("--outdir", default="results/analysis", help="Output folder.")
    ap.add_argument("--mode", choices=["classification", "regression"], default="classification")
    ap.add_argument("--label-col", default="label", help="Label column for classification.")
    ap.add_argument("--no-focus-colon", action="store_true", help="Do not filter to colon cancer.")
    ap.add_argument("--target", default="last_days_to_follow_up", help="Main target for regression.")
    ap.add_argument(
        "--extra-targets",
        default="mean_age_at_dx,bmi_last,weight_last,height_last",
        help="Comma-separated extra targets to try.",
    )
    ap.add_argument("--max-missing", type=float, default=0.6, help="Max missing rate for targets.")
    ap.add_argument("--min-rows", type=int, default=80, help="Min rows to run a regression.")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test size for regression.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--max-unique-cat", type=int, default=100, help="Max unique categories to keep.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    df = pd.read_csv(csv_path)

    sns.set_theme(style="whitegrid")

    df, dropped_age_cols = _convert_age_days_to_years(df)
    if dropped_age_cols:
        df = df.drop(columns=dropped_age_cols, errors="ignore")

    focus_colon = not bool(args.no_focus_colon)
    colon_info = {"filtered": 0, "input_rows": int(len(df)), "output_rows": int(len(df))}
    if focus_colon:
        df, colon_info = _filter_colon_cases(df)

    miss_df = _save_missingness(df, outdir)
    _write_summary(df, miss_df, outdir)

    if args.mode == "classification":
        label_col = args.label_col
        prefix = "colon" if focus_colon else "all"
        _plot_label_distribution(df, label_col, outdir, prefix=prefix)
        _plot_label_rate_by_group(df, label_col, "sex", outdir, prefix=prefix)
        _plot_label_rate_by_group(df, label_col, "disease_type", outdir, prefix=prefix)
        _plot_label_rate_by_group(df, label_col, "primary_site", outdir, prefix=prefix)
        _plot_label_rate_by_group(df, label_col, "ethnicity", outdir, prefix=prefix)
        _plot_corr_heatmap(df, outdir, prefix=prefix)
        for col in ("age_years", "mean_age_at_dx_years", "bmi_last", "weight_last", "height_last"):
            if col in df.columns:
                _plot_box_by_label(df, col, outdir, prefix=prefix)
                _plot_hist_by_label(df, col, outdir, prefix=prefix)

        metrics = run_classification(
            df,
            label_col=label_col,
            outdir=outdir,
            prefix=prefix,
            test_size=float(args.test_size),
            seed=int(args.seed),
            max_missing=0.95,
            max_unique_cat=int(args.max_unique_cat),
        )
        metrics["focus_colon"] = int(focus_colon)
        metrics["colon_input_rows"] = int(colon_info["input_rows"])
        metrics["colon_output_rows"] = int(colon_info["output_rows"])
        metrics_json = _prefix_name(prefix, "classification_metrics.json")
        metrics_csv = _prefix_name(prefix, "classification_metrics.csv")
        with (outdir / metrics_json).open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame([metrics]).to_csv(outdir / metrics_csv, index=False)
    else:
        extra_targets = [c.strip() for c in args.extra_targets.split(",") if c.strip()]
        targets = _select_targets(
            df,
            main_target=args.target,
            extra_targets=extra_targets,
            max_missing=float(args.max_missing),
            min_rows=int(args.min_rows),
        )

        metrics_rows: List[Dict[str, float]] = []
        for target in targets:
            _plot_target_hist(df, target, outdir)
            _plot_box_by_label(df, target, outdir)
            _plot_group_means(df, target, "sex", outdir)
            _plot_group_means(df, target, "disease_type", outdir)
            _plot_group_means(df, target, "primary_site", outdir)
            _plot_group_means(df, target, "ethnicity", outdir)
            _plot_top_correlations(df, target, outdir)

            metrics = run_regression(
                df,
                target=target,
                outdir=outdir,
                test_size=float(args.test_size),
                seed=int(args.seed),
                max_missing=0.95,
                max_unique_cat=int(args.max_unique_cat),
            )
            metrics_rows.append(metrics)

        if metrics_rows:
            pd.DataFrame(metrics_rows).to_csv(outdir / "regression_metrics.csv", index=False)


if __name__ == "__main__":
    main()
