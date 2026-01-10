# Model evaluation summary

Data: combined_aligned.csv with deduplication to match training preprocessing.
Rows before dedupe: 42363
Rows after dedupe: 24096

Cross-validation (5-fold) results: see cv_results.csv and cv_summary.csv.
Another split (70/15/15, random_state=1337): see split_eval.csv.
Threshold scans per model are saved as threshold_scan_<model>.csv.
External validation: not available with current data sources (no independent cohort).