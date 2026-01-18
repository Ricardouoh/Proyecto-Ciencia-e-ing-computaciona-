from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def parse_anchor_string(value: str) -> List[Tuple[float, float]]:
    if not value:
        return []
    anchors: List[Tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        age_s, prob_s = item.split(":", 1)
        anchors.append((float(age_s), float(prob_s)))
    return anchors


def _normalize_percentiles(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        v = float(v)
        out.append(v / 100.0 if v > 1.0 else v)
    return out


def build_age_prior(anchors: List[Tuple[float, float]], mode: str = "floor", alpha: float = 0.5) -> Dict:
    if not anchors:
        return {}
    anchors_sorted = sorted(anchors, key=lambda x: x[0])
    return {
        "mode": mode,
        "alpha": float(alpha),
        "anchors": [[float(a), float(p)] for a, p in anchors_sorted],
    }


def build_age_prior_from_percentiles(
    ages: pd.Series,
    percentiles: Iterable[float],
    probs: Iterable[float],
    mode: str = "floor",
    alpha: float = 0.5,
) -> Dict:
    age_vals = pd.to_numeric(ages, errors="coerce").dropna()
    if age_vals.empty:
        return {}
    pct = _normalize_percentiles(percentiles)
    probs_list = [float(p) for p in probs]
    if len(pct) != len(probs_list):
        raise ValueError("percentiles y probs deben tener el mismo largo.")
    q = np.nanpercentile(age_vals.to_numpy(), [p * 100 for p in pct])
    anchors = list(zip(q.tolist(), probs_list))
    return build_age_prior(anchors, mode=mode, alpha=alpha)


def load_age_prior(path: str | Path) -> Dict:
    prior_path = Path(path)
    if not prior_path.exists():
        return {}
    try:
        data = json.loads(prior_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict) and data.get("anchors"):
        return data
    return {}


def save_age_prior(prior: Dict, path: str | Path) -> None:
    if not prior:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prior, indent=2), encoding="utf-8")


def age_prior_prob(ages: pd.Series, anchors: List[List[float]]) -> np.ndarray:
    if not anchors:
        return np.array([])
    anchor_ages = np.array([a[0] for a in anchors], dtype=float)
    anchor_probs = np.array([a[1] for a in anchors], dtype=float)
    age_vals = pd.to_numeric(ages, errors="coerce").to_numpy()
    probs = np.interp(age_vals, anchor_ages, anchor_probs, left=anchor_probs[0], right=anchor_probs[-1])
    return probs


def apply_age_prior(proba: np.ndarray, ages: pd.Series, prior: Dict) -> np.ndarray:
    if not prior or proba is None or ages is None:
        return proba
    anchors = prior.get("anchors", [])
    if not anchors:
        return proba
    mode = str(prior.get("mode", "floor")).lower()
    alpha = float(prior.get("alpha", 0.5))
    p_age = age_prior_prob(ages, anchors)
    p_age = np.nan_to_num(p_age, nan=0.0)
    if mode == "blend":
        return (1.0 - alpha) * proba + alpha * p_age
    if mode == "odds":
        eps = 1e-6
        p = np.clip(proba, eps, 1 - eps)
        pa = np.clip(p_age, eps, 1 - eps)
        logit = np.log(p / (1 - p)) + alpha * np.log(pa / (1 - pa))
        return 1.0 / (1.0 + np.exp(-logit))
    return np.maximum(proba, p_age)


def clip_proba(proba: np.ndarray, max_proba: float) -> np.ndarray:
    if proba is None:
        return proba
    if max_proba is None:
        return proba
    try:
        cap = float(max_proba)
    except Exception:
        return proba
    if cap <= 0 or cap >= 1:
        return proba
    return np.minimum(proba, cap)
