# Informe del modelo (PU + calibracion)

## Resumen ejecutivo
- Modelo: Gradient Boosting (GBR) con calibracion (sigmoid) y enfoque PU.
- Umbral operativo: 0.50 (manual) para mantener selectividad tras reforzar edad.
- Peso de edad: `--age-weight 3.0` y bandas de edad sin normalizar.
- Brier score (val/test): 0.1371 / 0.1374.

## Datos usados
- Fuente: `data/processed_smoke/`
- Tamaños:
  - Train: 23403 filas (aligned_train.csv)
  - Val: 5015 filas (aligned_val.csv)
  - Test: 5016 filas (aligned_test.csv)

## Umbral y por que se eligio 0.50
El umbral 0.50 se fija manualmente porque:
- Se aumento el peso de edad (`--age-weight 3.0`) y las bandas de edad con meseta alta.
- Esto eleva las probabilidades en pacientes mayores, por lo que un umbral mas alto evita falsos positivos excesivos.
- Se mantiene un recall alto sin perder selectividad global.

## Metricas resultantes (PU + calibracion)
Validacion (metrics_val.json):
- AUROC: 0.8714
- AUPRC: 0.8788
- Accuracy: 0.8435
- Precision: 0.8125
- Recall: 0.9836
- F1: 0.8899
- Threshold: 0.50 (n_val=5015)

Test (test_metrics.json):
- AUROC: 0.8767
- AUPRC: 0.8845
- Accuracy: 0.8417
- Precision: 0.8144
- Recall: 0.9764
- F1: 0.8881
- Threshold: 0.50 (n_test=5016)

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
- Val: 0.1371
- Test: 0.1374
Interpretacion: el error promedio de probabilidad es moderado; refleja el refuerzo agresivo de edad y el cap de probabilidad.

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
