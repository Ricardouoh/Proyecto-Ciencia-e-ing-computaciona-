from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def load_feature_weights(path: str | Path) -> Dict[str, float]:
    weight_path = Path(path)
    if not weight_path.exists():
        return {}
    try:
        data = json.loads(weight_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        weights = data.get("weights", data)
        if isinstance(weights, dict):
            return {str(k): float(v) for k, v in weights.items()}
    return {}


def save_feature_weights(weights: Dict[str, float], path: str | Path) -> None:
    if not weights:
        return
    weight_path = Path(path)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": "substring", "weights": weights}
    weight_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def apply_feature_weights(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    if not weights:
        return df
    out = df.copy()
    for pattern, weight in weights.items():
        cols = [c for c in out.columns if pattern in c]
        if not cols:
            continue
        out[cols] = out[cols] * float(weight)
    return out
