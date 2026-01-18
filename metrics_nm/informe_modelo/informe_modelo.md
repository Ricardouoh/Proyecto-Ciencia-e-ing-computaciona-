# Informe del modelo (nuevo modelo)

## Datos usados y split
- Dataset final: `data/processed_raw_full/`
- Train: 23403 filas (`aligned_train.csv`)
- Val: 5015 filas (`aligned_val.csv`)
- Test: 5016 filas (`aligned_test.csv`)

La division fue 70/15/15 con estratificacion por label.

## Umbral operativo y justificacion
- Umbral seleccionado: 0.15
- Metrica de seleccion: F1
- Restricciones:
  - min_threshold: 0.10
  - max_threshold: 0.50
  - min_recall: 0.80
  - domain_penalty_weight: 0.3
  - PU target pos rate (NHANES): 0.12 (peso 0.5)

Se eligio 0.15 porque maximiza F1 bajo estas restricciones y mantiene la tasa de positivos en NHANES cerca del objetivo, evitando soluciones triviales.

## Metricas del modelo (PU + calibracion)
Validacion (metrics_val.json):
- AUROC: 0.9915
- AUPRC: 0.9957
- Accuracy: 0.9531
- Precision: 0.9603
- Recall: 0.9671
- F1: 0.9637

Test (test_metrics.json):
- AUROC: 0.9907
- AUPRC: 0.9951
- Accuracy: 0.9454
- Precision: 0.9395
- Recall: 0.9780
- F1: 0.9584
- Threshold: 0.15

## Que significa cada metrica
- AUROC: capacidad de separar clases en todos los umbrales.
- AUPRC: precision vs recall (clave con clases desbalanceadas).
- Accuracy: proporcion total de aciertos.
- Precision: de los positivos predichos, cuantos son correctos.
- Recall: de los positivos reales, cuantos fueron detectados.
- F1: balance entre precision y recall.

## Brier score
- Brier score: 0.03636
Indica calidad de la probabilidad. Menor es mejor; este valor sugiere buena calibracion.

## SHAP
SHAP (Shapley values) explica la contribucion de cada variable a la probabilidad de cada paciente.
Se usa en:
- `src/predict.py` para explicar factores que suben o bajan el riesgo por paciente.
- `src/compare_shap.py` para comparar pacientes y detectar el punto de quiebre.

## PU Learning (Positive-Unlabeled)
NHANES se trata como no etiquetado (no 100% sano).
Se aplico:
- label smoothing: 0.05
- objetivo de tasa de positivos en NHANES: 0.12

Efecto:
- Evita colapso del umbral a 0.0 o 1.0.
- Mejora estabilidad del score y mantiene alto recall en positivos confirmados.

## Artefactos del modelo actual
- `results/model.joblib`
- `results/preprocessor.joblib`
- `results/threshold.json`
- `results/metrics_val.json`
- `results/test_metrics.json`
- `results/threshold_scan.csv`
- `results/final_scientific_report.json`
