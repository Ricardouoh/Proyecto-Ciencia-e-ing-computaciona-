Calibration comparison with fixed bins (10 equal-width bins)

Key deltas
- Brier score: 0.02413841903820455 -> 0.023373596912875733
- Mean abs error (bins present): 0.11002970149646121 -> 0.0730067989607074
- Max abs error (bins present): 0.34043515257069235 -> 0.3030013855537639
- Mean signed error: 0.08213682508878581 -> 0.05696031994342619

Notes
- Both curves use the same bin edges from 0.0 to 1.0.
- Test split reproduced with seed=42 and dedupe; see calibration_fixed_bins_comparison.csv for per-bin details.