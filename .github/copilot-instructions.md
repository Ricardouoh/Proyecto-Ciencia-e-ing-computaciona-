**Repository Overview**
- **Purpose**: Small ML pipeline for tabular clinical data (preprocessing → train → evaluate → lightweight FastAPI inference).
- **Key directories**: `src/` (core code), `data/raw.csv` and `data/processed/` (data artifacts), `results/` (model & preprocessor by convention).

**Big Picture Architecture**
- **Data ingestion & normalization**: `src/data_loader.py` — loads CSV/JSONL/Parquet/XLSX, normalizes column names to `snake_case`, optional drops (constant, duplicate columns/rows). Example CLI: `python -m src.data_loader --in data/raw_folder --out data/raw.csv`.
- **Preprocessing pipeline**: `src/preprocess.py` — cleaning (drop id-like, high-NaN, high-cardinality), stratified split, imputation, numeric scaling and one-hot encoding. ColumnTransformer uses transformer names `num` and `cat` (important for downstream code that introspects the preprocessor). Default CLI example: `python -m src.preprocess --csv data/raw.csv --target label --outdir data/processed`.
- **Model definition**: `src/model.py` — exposes `make_model(cfg)` and `default_config(model)` supporting `logreg` (LogisticRegression) and `mlp` (MLPClassifier). Use these factories rather than instantiating sklearn classes directly.
- **Training loop & artifacts**: `src/train_loop.py` — `train_and_validate(cfg, data_dir)` implements training with manual early stopping for MLP (warm_start strategy). Training writes `results/model.joblib` and a `results/train_log.csv` by default.
- **Evaluation & metrics**: `src/evaluate.py` + `src/metrics.py` — compute AUROC/AUPRC, thresholded metrics and save ROC/PR plots and JSON metrics. Helpers: `compute_classification_metrics`, `plot_roc`, `plot_pr`, `save_json`.
- **Inference API**: a FastAPI app exists in code (see `src/train_loop.py` top-level FastAPI bits and `src/api.py` header docs). The API expects the preprocessor and model artifacts and exposes `/health`, `/schema`, `/predict`.

**Important Conventions & Gotchas (project-specific)**
- **Column naming**: `data_loader` enforces `snake_case` for raw CSV columns. Upstream tools assume this normalization.
- **Transformer names**: The preprocessor ColumnTransformer uses transformer names `num` and `cat`. Code that inspects available raw columns for inference relies on those names.
- **Preprocessor persistence paths**: there is a mismatch to be aware of:
  - `src/preprocess.py` CLI saves the preprocessor to the `--outdir` (default `data/processed/preprocessor.joblib`).
  - `src/train_loop.py` and the FastAPI code expect `results/preprocessor.joblib` (see `PREPROC_PATH = Path("results/preprocessor.joblib")`).
  - When running end-to-end, either move or copy the preprocessor to `results/` or pass/modify paths consistently.
- **Model artifact**: canonical model path used by inference is `results/model.joblib` — training saves best checkpoint there.
- **One-hot feature names**: after fit, `OneHotEncoder.get_feature_names_out(categorical_cols)` is used to build final column names. Downstream code expects these deterministic names.

**Common CLI flows / commands**
- Unify raw files into a single CSV:
  - `python -m src.data_loader --in path/to/files --out data/raw.csv --target label`
- Preprocess (clean, split, fit preprocessor, save CSVs and preprocessor):
  - `python -m src.preprocess --csv data/raw.csv --target label --outdir data/processed`
- Train (from Python; there is no top-level `train` CLI):
  - Minimal python invocation example:
    - `python -c "from src.train_loop import TrainConfig, train_and_validate; from pathlib import Path; cfg=TrainConfig(model_cfg={'model':'mlp','mlp':{'max_iter':1}}); train_and_validate(cfg, Path('data/processed'))"`
  - Note: training writes best model to `results/model.joblib` by convention.
- Evaluate model on test set (helper functions available):
  - Use `src.evaluate` functions or call `evaluate_model(model_path=Path('results/model.joblib'), test_csv=Path('data/processed/test.csv'), outdir=Path('results/eval'))` from Python.

**Patterns an AI agent should follow**
- Prefer using project factories: call `src.model.make_model(cfg)` and `src.preprocess.make_preprocessor(...)` to keep parameter handling consistent.
- When inspecting available raw features for inference, read the saved preprocessor and look for transformers named `num` / `cat` (see `_infer_raw_columns_from_preprocessor` in `src/train_loop.py`).
- Use saved artifacts paths (`results/*`) unless a different path is explicitly passed. If a script saves to `data/processed/`, ensure to copy/move the preprocessor to `results/` before starting the FastAPI app.
- For MLP training the code uses `max_iter=1` + `warm_start=True` to implement epoch-based training. Respect that pattern if you modify training logic.

**Files to inspect when editing behavior**
- `src/data_loader.py` — input normalization, encodings, CLI options.
- `src/preprocess.py` — cleaning heuristics (ID-like dropping, NaN thresholds, high-cardinality filtering), split logic, and how preprocessor columns are built.
- `src/model.py` — model parameter defaults and supported model list.
- `src/train_loop.py` — training loop, early stopping logic, artifact paths, and the inference schema helper.
- `src/metrics.py` — metrics computation and plotting utilities.

**If you need to run tests or debug**
- There are no automated tests in the repo. Use small, local runs using the CLIs above and inspect `results/` and `data/processed/` artifacts.
- For the FastAPI inference server (local dev): ensure `results/preprocessor.joblib` and `results/model.joblib` exist, then run a small ASGI server, for example:
  - `uvicorn src.train_loop:app --reload --port 8000`  (the FastAPI `app` lives in `src/train_loop.py` in this repo)

**What I preserved / merging note**
- No existing `.github/copilot-instructions.md` was found; this file is a fresh project-specific guide.

Please review for accuracy and tell me which sections you want expanded (examples, exact train invocation, CI hooks, or preferred artifact locations to standardize).
