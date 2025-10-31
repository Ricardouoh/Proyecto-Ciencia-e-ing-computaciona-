from __future__ import annotations
"""
API de inferencia con FastAPI.

Endpoints:
- GET  /health        -> estado del servicio
- GET  /schema        -> columnas crudas esperadas por el preprocesador
- POST /predict       -> probabilidad y clase (umbral configurable)

Requisitos (artefactos entrenados):
- results/model.joblib
- results/preprocessor.joblib
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
