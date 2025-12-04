from __future__ import annotations
"""
Evaluación del mejor modelo en el conjunto de test.

- Carga:
    * results/model.joblib
    * data/processed/test.csv
- Calcula:
    * AUROC
    * AUPRC
    * accuracy
- Guarda:
    * results/test_metrics.json
    
    
    
    
SALIDA DEL MÓDULO DE EVALUACIÓN:

1) Métricas impresas por consola:

   - AUROC (Area Under the ROC Curve):
     Mide la capacidad del modelo para distinguir entre pacientes de la clase
     positiva y la clase negativa en todos los posibles umbrales de decisión.
     Valores cercanos a 1 indican excelente capacidad de discriminación.
     Un valor cercano a 0.5 indica un modelo sin poder predictivo.

   - AUPRC (Area Under the Precision-Recall Curve):
     Evalúa el compromiso entre precisión y exhaustividad (recall),
     especialmente útil cuando las clases están desbalanceadas.
     Valores altos indican que el modelo identifica correctamente los
     casos positivos con pocos falsos positivos.

   - Accuracy (Exactitud):
     Porcentaje total de predicciones correctas realizadas por el modelo
     sobre el conjunto de prueba.
     Puede ser engañosa en conjuntos de datos desbalanceados, por lo que
     debe interpretarse junto con AUROC y AUPRC.

   - n_test:
     Cantidad de muestras utilizadas en la evaluación final del modelo.

2) Archivo opcional: results/test_metrics.json

   Contiene:
   - El valor del umbral utilizado para clasificar (threshold)
   - Las métricas AUROC, AUPRC y Accuracy
   - El número total de muestras evaluadas (n_test)
   Este archivo permite conservar resultados para reportes, gráficos y
   comparación entre distintos modelos entrenados.

------------------------------------------------------------

INTERPRETACIÓN DE LOS RESULTADOS:

Las métricas obtenidas en esta etapa reflejan el DESEMPEÑO REAL del modelo
en un escenario clínico simulado, ya que:

- El conjunto de prueba (test.csv) nunca fue utilizado en el proceso de
  entrenamiento ni durante la etapa de validación.
- Esto evita el sobreajuste artificial y permite estimar la capacidad real
  de generalización del modelo sobre pacientes no vistos.
- Valores altos de AUROC y AUPRC indican que el modelo presenta un alto poder
  predictivo y una excelente capacidad de discriminación clínica.
- Si las métricas fueran bajas, significaría que el modelo es inestable,
  poco generalizable o que requiere mejorar el preprocesamiento,
  la arquitectura o el balance de clases.

En resumen, estas métricas representan la calidad final del sistema de
predicción clínica construido en el proyecto.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)


def _load_xy(csv_path: Path, target: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"No se encuentra la columna {target} en {csv_path}")
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y


def evaluate_test(
    model_path: str | Path = "results/model.joblib",
    test_csv: str | Path = "data/processed/test.csv",
    out_json: str | Path = "results/test_metrics.json",
) -> Dict:
    model_path = Path(model_path)
    test_csv = Path(test_csv)
    out_json = Path(out_json)

    if not model_path.exists():
        raise FileNotFoundError(f"No encontré el modelo en {model_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"No encontré el test CSV en {test_csv}")

    model = joblib.load(model_path)
    Xte, yte = _load_xy(test_csv, target="label")

    proba = model.predict_proba(Xte)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    metrics = {}
    try:
        metrics["auroc"] = roc_auc_score(yte, proba)
    except Exception:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = average_precision_score(yte, proba)
    except Exception:
        metrics["auprc"] = float("nan")

    metrics["accuracy"] = accuracy_score(yte, y_pred)
    metrics["n_test"] = int(len(yte))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("===== MÉTRICAS EN TEST =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("============================")

    return metrics


if __name__ == "__main__":
    evaluate_test()
