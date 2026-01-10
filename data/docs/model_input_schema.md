Model input schema (calibrated model)

Use this CSV schema for inference. Do NOT include the label column.

Required columns (raw, before preprocessing)
- age_years: float, years (range seen ~0.4 to 90)
- sex: category string (use lowercase values listed below)
- ethnicity: category string (use lowercase values listed below)
- race: category string (case sensitive for "Unknown")
- height_last: float, cm
- weight_last: float, kg
- bmi_last: float
- tobacco_smoking_status_any: 0 or 1

Allowed category values (from training data)
- sex: female | male | not reported | unknown
- ethnicity: hispanic or latino | not hispanic or latino | not reported | unknown
- race: Unknown | american indian or alaska native | asian | black or african american | not reported | other | white

Notes
- If a value is missing, leave it empty or use "unknown"/"not reported".
- Unknown categories are ignored by the encoder (no error), but can reduce signal.
- The model outputs a probability for label=1 (cancer). Use threshold 0.5 unless you choose another.
- Extra columns are allowed; only the required columns are used.
- tobacco_smoking_status_any can be 0/1 or strings like current/former/never/yes/no.

Example file
- data/docs/model_input_template.csv
