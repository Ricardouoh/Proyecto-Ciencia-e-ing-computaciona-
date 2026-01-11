from __future__ import annotations
"""
Definicion de modelos:
- Regresion Logistica (baseline)
- MLPClassifier (red densa con backprop)
- GradientBoosting (clasificacion o regresion segun task)
- RandomForest (clasificacion o regresion segun task)
Uso:
    from src.model import make_model, default_config
    cfg = default_config(model="mlp")  # o "logreg"
    model = make_model(cfg, task="classification")
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_val)[:, 1]
"""

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPClassifier


def default_config(model: str = "mlp") -> Dict[str, Any]:
    """
    Config por defecto para cada modelo.
    - model: "logreg" | "mlp" | "gbr" | "rf"
    """
    if model == "logreg":
        return {
            "model": "logreg",
            "logreg": {
                "max_iter": 200,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "n_jobs": None,  # usa todos si None y solver lo permite
                "random_state": 42,
            },
        }
    if model == "mlp":
        return {
            "model": "mlp",
            "mlp": {
                # Arquitectura aún más ligera para mayor estabilidad en este dataset
                "hidden_layer_sizes": [32, 16],
                "activation": "relu",
                "solver": "adam",
                "alpha": 1e-4,               # L2
                "batch_size": "auto",
                "learning_rate": "adaptive",  # adapta LR si se estanca
                "learning_rate_init": 3e-4,
                "max_iter": 100,
                "early_stopping": True,
                "n_iter_no_change": 10,       # paciencia (val)
                "validation_fraction": 0.1,   # solo lo usa internamente el MLP
                "shuffle": True,
                "random_state": 42,
                "verbose": False,
            },
        }
    if model == "gbr":
        return {
            "model": "gbr",
            "gbr": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 0.9,
                "random_state": 42,
            },
        }
    if model == "rf":
        return {
            "model": "rf",
            "rf": {
                "n_estimators": 300,
                "max_depth": 5,
                "min_samples_leaf": 5,
                "random_state": 42,
                "n_jobs": -1,
            },
        }
    raise ValueError("Modelo no soportado. Usa 'logreg', 'mlp', 'gbr' o 'rf'.")


def make_model(cfg: Dict[str, Any], task: str = "classification"):
    """
    Crea el estimador sklearn según cfg.
    cfg ejemplo:
    {
        "model": "mlp",
        "mlp": {
            "hidden_layer_sizes": [128, 64],
            "max_iter": 100,
            "early_stopping": True,
            ...
        }
    }
    """
    model_name = cfg.get("model", "mlp").lower()

    if model_name == "logreg":
        p = (cfg.get("logreg") or {}).copy()
        # Defaults seguros si faltan
        p.setdefault("max_iter", 200)
        p.setdefault("class_weight", "balanced")
        p.setdefault("solver", "lbfgs")
        p.setdefault("n_jobs", None)
        p.setdefault("random_state", 42)
        return LogisticRegression(**p)

    if model_name == "mlp":
        p = (cfg.get("mlp") or {}).copy()
        # Defaults seguros si faltan
        p.setdefault("hidden_layer_sizes", [128, 64])
        p.setdefault("activation", "relu")
        p.setdefault("solver", "adam")
        p.setdefault("alpha", 1e-4)
        p.setdefault("batch_size", "auto")
        p.setdefault("learning_rate", "adaptive")
        p.setdefault("learning_rate_init", 1e-3)
        p.setdefault("max_iter", 100)
        p.setdefault("early_stopping", True)
        p.setdefault("n_iter_no_change", 10)
        p.setdefault("validation_fraction", 0.1)
        p.setdefault("shuffle", True)
        p.setdefault("random_state", 42)
        p.setdefault("verbose", False)

        # Acepta listas para hidden_layer_sizes
        if isinstance(p.get("hidden_layer_sizes"), list):
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])

        return MLPClassifier(**p)

    if model_name == "gbr":
        p = (cfg.get("gbr") or {}).copy()
        p.setdefault("n_estimators", 200)
        p.setdefault("learning_rate", 0.05)
        p.setdefault("max_depth", 3)
        p.setdefault("subsample", 0.9)
        p.setdefault("random_state", 42)
        if task == "classification":
            return GradientBoostingClassifier(**p)
        return GradientBoostingRegressor(**p)

    if model_name == "rf":
        p = (cfg.get("rf") or {}).copy()
        p.setdefault("n_estimators", 300)
        p.setdefault("max_depth", 5)
        p.setdefault("min_samples_leaf", 5)
        p.setdefault("random_state", 42)
        p.setdefault("n_jobs", -1)
        if task == "classification":
            return RandomForestClassifier(**p)
        return RandomForestRegressor(**p)

    raise ValueError("Modelo no soportado. Usa 'logreg', 'mlp', 'gbr' o 'rf'.")


def get_supported_models() -> Dict[str, str]:
    """Devuelve un mapa simple de modelos soportados."""
    return {
        "logreg": "LogisticRegression (baseline, lineal)",
        "mlp": "MLPClassifier (red densa con backprop, no lineal)",
        "gbr": "GradientBoosting (no lineal, clasificacion o regresion)",
        "rf": "RandomForest (no lineal, clasificacion o regresion)",
    }
