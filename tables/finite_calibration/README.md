# Finite-calibration reader data

These tables describe the calibration-size intervention. D and R
denote DeBERTa-v3-base and RoBERTa-base. Calibration sizes are 64/128/256/512;
T1--T5 use PCG64 permutation seeds 202610401--202610405 and nested prefixes.

`raw_kl_complete.csv` contains 1170 unique actual endpoints: 1080 reduced-N
candidates and 90 unique archived N512 candidates. `rank_raw_kl_display.csv`
contains 1440 display rows, including explicit aliases of the reused M0
endpoint. Aliases are not additional inference or independent observations.
`coefficient_stability_complete.csv` contains 1152 unique method conditions;
M0 is independent of calibration geometry. `estimator_scaling.csv` contains 176
estimates across all 11 sources, with one unique N512 row per source.

Reduced-N means, sample SDs, medians and ranges describe five fixed nested
subsets within a source. N512 is one unique full-pool observation and its
reported subset SD=0 is a display convention, not uncertainty estimation.
There are no new bootstrap intervals. All adverse, null and nonmonotonic
results, including RoBERTa607/809, are retained. The regime/oracle quantities
are descriptive held-out summaries and are unavailable at deployment.
