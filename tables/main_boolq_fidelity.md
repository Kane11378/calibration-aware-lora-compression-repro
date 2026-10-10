**BoolQ fidelity and source quality. Means weight ranks 8/4/2 equally. Negative H1--M0 differences favor H1; intervals are the stored paired 95% intervals.**

| Source | M0 KL | H1 KL | S1 KL | H1--M0 [95% CI] | Reduction | Acc. (%) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **Panel A: DeBERTa-v3-base** |  |  |  |  |  |  |
| 307 | 0.025316 | 0.017437 | 0.016773 | -0.007879 [-0.009969, -0.005964] | 31.12% | 85.29 |
| 401 | 0.030893 | 0.022633 | 0.021472 | -0.008260 [-0.009982, -0.006622] | 26.74% | 85.54 |
| 503 | 0.015626 | 0.006974 | 0.006935 | -0.008652 [-0.009763, -0.007618] | 55.37% | 83.98 |
| **Panel B: RoBERTa-base** |  |  |  |  |  |  |
| 607† | 2.215080e-7 | 8.102366e-8 | 8.122970e-8 | -1.404843e-7 [-1.467954e-7, -1.343416e-7] | 63.42% | 62.17 |
| 701 | 0.036523 | 0.018071 | 0.018235 | -0.018452 [-0.020373, -0.016558] | 50.52% | 79.33 |
| 809 | 0.030045 | 0.027901 | 0.026011 | -0.002144 [-0.002430, -0.001866] | 7.14% | 80.61 |

† Seed 607 predicts class 1 on all 3,270 inputs (0/3,270 class 0/class 1 predictions), exactly matching 62.17% majority prevalence. Its 63.42% reduction has negligible absolute magnitude. Complete class counts for every source are in tables/source_quality_boolq.md.
