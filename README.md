# Proyecto de screening de cancer (HCMI/TCGA + NHANES)

Pipeline de mineria de datos y ML clinico para estimar riesgo de cancer con enfoque en
recall, calibracion probabilistica y explicabilidad (SHAP). El sistema entrena con
casos positivos confirmados (HCMI/TCGA) y poblacion general no etiquetada (NHANES)
usando enfoque PU (Positive-Unlabeled).

## Caracteristicas clave
- Alineacion de dominios y esquema canonico de variables clinicas.
- Mitigacion de fuga de dominio (variable blinding).
- Entrenamiento con pesos de riesgo y PU label smoothing.
- Calibracion de probabilidades (sigmoid).
- Reportes de metricas, Brier score e interpretabilidad SHAP.
- Scripts para inferencia clinica y comparacion de pacientes.

## Requisitos
- Python 3.11+
- Dependencias principales: numpy, pandas, scikit-learn, joblib, matplotlib, seaborn
- SHAP (opcional para explicaciones): shap

Ejemplo de instalacion:
```bash
pip install numpy pandas scikit-learn joblib matplotlib seaborn shap
```

## Estructura del repositorio (resumen)
```
data/
  clinical/
    New_cancer/                # JSON clinicos HCMI/TCGA
    reextracted/               # Tablas reextraidas desde JSON
      raw/                     # raw_hcmi_tcga.csv
  nhanes/
    nhanes_merged.csv          # NHANES consolidado
  processed_raw_full/          # aligned_train/val/test.csv (legacy)
  processed_smoke/             # aligned_train/val/test.csv (dataset final actual)

src/
  Build dataset/               # scripts de armado de dataset
    reextract_clinical_tables.py
    build_raw_clinical.py
    align_datasets.py
  train_loop.py                # entrenamiento
  summarize_training.py        # metricas y resumen
  final_scientific_report.py   # reportes finales
  predict.py                   # inferencia clinica + SHAP
  compare_shap.py              # comparacion SHAP entre pacientes

results/                       # artefactos del modelo actual
metrics_nm/informe/            # informe en texto del modelo
```

## Flujo recomendado (end-to-end)

## Etapas de datos y limpieza
1) Extraccion clinica (JSON -> tablas):
   - Script: `src/Build dataset/reextract_clinical_tables.py`
   - Resultado: tablas planas por proyecto en `data/clinical/reextracted/`
   - Limpieza: normalizacion de tokens faltantes (not reported/unknown -> NaN)

2) Construccion dataset crudo HCMI/TCGA:
   - Script: `src/Build dataset/build_raw_clinical.py`
   - Resultado: `data/clinical/reextracted/raw/raw_hcmi_tcga.csv`
   - Limpieza/ingenieria:
     - Calculo de `age_years` desde edad/dias al nacimiento o diagnosico.
     - Agregacion de variables antropometricas (ultimo BMI/height/weight).
     - Normalizacion de tabaquismo y pack-years (si aplica).

3) NHANES consolidado:
   - Archivo base: `data/nhanes/nhanes_merged.csv`
   - Este archivo ya contiene variables clinicas combinadas (edad, sexo, raza, BMI).

4) Alineacion y esquema canonico:
   - Script: `src/Build dataset/align_datasets.py`
   - Normaliza categorias (sex/ethnicity/race), convierte numericos y crea columnas faltantes.
   - Mitiga fuga de dominio eliminando variables de tabaquismo (`--drop-features ...`).
   - Resultado final actual: `data/processed_smoke/aligned_train.csv`, `aligned_val.csv`, `aligned_test.csv`.

5) Split de train/val/test:
   - Se usa estratificacion por label con 70/15/15.
   - La division ocurre dentro de `align_datasets.py` y queda guardada en `data/processed_raw_full/`.

6) Preprocesamiento para entrenamiento:
   - En `src/train_loop.py` se ajusta el preprocesador:
     - Imputacion de NaN.
     - Escalado numerico.
     - One-hot encoding de categoricas.
   - Se guarda en `results/preprocessor.joblib`.

### 1) Reextraer tablas clinicas (JSON -> CSV)
```bash
python "src/Build dataset/reextract_clinical_tables.py" ^
  --input-dir data/clinical/New_cancer ^
  --outdir data/clinical/reextracted ^
  --reset-output
```

### 2) Construir dataset crudo HCMI/TCGA
```bash
python "src/Build dataset/build_raw_clinical.py" ^
  --input-dir data/clinical/reextracted ^
  --outdir data/clinical/reextracted/raw ^
  --label 1 --domain hcmi_tcga
```

Salida principal:
- `data/clinical/reextracted/raw/raw_hcmi_tcga.csv`

### 3) Alinear HCMI/TCGA y NHANES (esquema canonico)
```bash
python "src/Build dataset/align_datasets.py" ^
  --hcmi data/clinical/reextracted/raw/raw_hcmi_tcga.csv ^
  --nhanes data/nhanes/nhanes_merged.csv ^
  --outdir data/processed_raw_full ^
  --drop-features tobacco_smoking_status_any,pack_years_smoked_max ^
  --blind-mode drop
```

Salida principal:
- `data/processed_raw_full/aligned_train.csv`
- `data/processed_raw_full/aligned_val.csv`
- `data/processed_raw_full/aligned_test.csv`

### 4) Entrenar modelo (PU + calibracion)
```bash
python -m src.train_loop ^
  --data-dir data/processed_smoke ^
  --model gbr ^
  --optimize-threshold --metric f1 ^
  --min-threshold 0.1 --max-threshold 0.5 --min-recall 0.8 ^
  --age-weight 3.0 ^
  --age-band-weights "<30:0.1,30-45:0.6,45-55:2.0,55-75:6.0,>75:8.0" ^
  --no-age-band-normalize ^
  --max-proba 0.70 ^
  --pu-label-smoothing 0.05 --pu-unlabeled-domain nhanes ^
  --pu-target-pos-rate 0.12 --pu-target-weight 0.5 ^
  --calibrate --calibration-method sigmoid ^
  --risk-weight --risk-weight-factor 0.5 --risk-weight-min-age 60 ^
  --risk-weight-use-race --domain-penalty-weight 0.3
```

### 5) Resumen y metricas
```bash
python -m src.summarize_training ^
  --data-dir data/processed_smoke ^
  --results-dir results ^
  --unlabeled-domain nhanes
```

### 6) Reporte final cientifico
```bash
python -m src.final_scientific_report ^
  --data-dir data/processed_smoke ^
  --results-dir results ^
  --metric brier --n-repeats 5 ^
  --age-ci-bootstrap 300 --high-risk-quantile 0.9 ^
  --unlabeled-domain nhanes
```

## Inference clinica

### Prediccion con explicaciones SHAP
```bash
python -m src.predict --input data/test1.csv --id-col patient_id
```

### Comparar SHAP entre dos pacientes
```bash
python -m src.compare_shap --input data/test1.csv --id-col patient_id --id-a 2 --id-b 3
```

### Nuevo test (prueba1.csv) con umbral 0.5
```bash
python -m src.predict --input data/tests/Pruebas_csv/prueba1.csv --id-col row_id --threshold 0.50
```
Explicacion: el umbral se sube a 0.50 para mantener selectividad luego de aumentar el peso de edad
(`--age-weight 3.0` + `--age-band-weights`), evitando que el prior de edad dispare demasiados positivos.

## Salidas importantes (results/)
- `results/model.joblib` y `results/preprocessor.joblib`
- `results/threshold.json` y `results/threshold_scan.csv`
- `results/metrics_val.json` y `results/test_metrics.json`
- `results/risk_concordance.json`
- `results/final_scientific_report.json`

## Notas tecnicas
- NHANES se trata como dominio no etiquetado (PU learning).
- Umbral operativo actual: 0.50 (manual) para controlar falsos positivos tras reforzar edad.
- SHAP se usa para explicar factores que suben o bajan el riesgo por paciente.

## Informe del modelo
Documento de lectura rapida en:
`metrics_nm/informe/informe_modelo.md`

## Licencia
Pendiente de definir.
