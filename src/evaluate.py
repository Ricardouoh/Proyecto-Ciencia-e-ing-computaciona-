from __future__ import annotations
"""
Evaluación en TEST:
- Carga results/model.joblib
- Lee data/processed/test.csv
- Calcula AUROC/AUPRC y métricas a umbral
- Guarda ROC.png, PR.png y test_metrics.json
"""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.metrics import (
    compute_classification_metrics,
    confusion_counts,
    plot_pr,
    plot_roc,
    save_json,
)
