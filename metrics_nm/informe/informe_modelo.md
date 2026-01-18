# Informe del modelo (PU + calibracion)

## Resumen ejecutivo
- Modelo: Gradient Boosting (GBR) con calibracion (sigmoid) y enfoque PU.
- Umbral operativo: 0.15 (optimizacion F1 con restricciones de recall y control de dominio).
- Brier score: 0.03636 (calidad de probabilidad).

## Datos usados
- Fuente: `data/processed_raw_full/`
- Tamaños:
  - Train: 23403 filas (aligned_train.csv)
  - Val: 5015 filas (aligned_val.csv)
  - Test: 5016 filas (aligned_test.csv)

## Umbral y por que se eligio 0.15
El umbral se eligio con:
- metric = F1
- min_threshold = 0.10, max_threshold = 0.50
- min_recall = 0.80
- domain_penalty_weight = 0.3 (reduce diferencias de tasa de positivos entre dominios)
- PU target: pos_rate NHANES ~0.12 con peso 0.5

El umbral 0.15 obtuvo el mejor F1 en validacion dentro de esas restricciones y mantuvo una tasa de positivos en NHANES cercana al objetivo (0.12). Esto evita soluciones triviales (todo positivo o todo negativo) y mantiene el recall alto.

## Metricas resultantes (PU + calibracion)
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
- Threshold: 0.15 (n_test=5016)

Nota: NHANES es dominio no etiquetado. Por eso el recall por dominio en NHANES no es interpretable como clasificacion supervisada; se reporta distribucion de riesgo en `results/risk_concordance.json`.

## Que significa cada metrica
- AUROC: capacidad de separar positivos y negativos en todos los umbrales.
- AUPRC: calidad en escenarios desbalanceados (precision vs recall).
- Accuracy: porcentaje total de aciertos (puede ser engañosa si hay desbalance).
- Precision: proporción de predicciones positivas que son correctas.
- Recall: proporción de positivos detectados (sensibilidad).
- F1: balance entre precision y recall.
- Brier score: error cuadratico medio de las probabilidades; menor es mejor.

## SHAP: que es y para que se uso
SHAP (Shapley values) cuantifica la contribucion de cada variable a la probabilidad individual.
En este proyecto se usa para:
- Explicar por paciente los factores que aumentan o disminuyen el riesgo.
- Comparar pacientes con resultados distintos y detectar el "punto de quiebre" entre Alto/Bajo riesgo.

Implementacion: `src/predict.py` (explicacion por paciente) y `src/compare_shap.py` (comparacion entre pacientes).

## Brier score del modelo
- Brier = 0.03636
Interpretacion: las probabilidades estan bien calibradas; el error promedio en probabilidades es bajo.

## PU Learning (Positive-Unlabeled)
NHANES se trata como no etiquetado (no todos son sanos). Para reducir sesgo:
- label smoothing en NHANES: 0.05
- objetivo de tasa de positivos en NHANES: 0.12

Efecto: evita colapso de umbral a 0.0, mejora la estabilidad de la probabilidad y mantiene recall alto en positivos confirmados (HCMI).

## Señales clinicas destacadas
Permutation importance (Brier):
1) ethnicity
2) age_years
3) race

Esto sugiere que la edad y variables demograficas son los principales moduladores del riesgo en el modelo actual.

## Archivos clave del modelo final
- `results/model.joblib`
- `results/preprocessor.joblib`
- `results/threshold.json`
- `results/metrics_val.json`
- `results/test_metrics.json`
- `results/risk_concordance.json`
- `results/final_scientific_report.json`
