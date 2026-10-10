**Task changes, H1 minus M0, with stored paired 95% intervals. CE is in nats; accuracy is in percentage points.**

| Source | Raw CE | Calibrated CE | Accuracy (pp) |
| :-- | :-- | :-- | :-- |
| DeBERTa 307 | +0.000496 [-0.004085, +0.005046] | -0.000638 [-0.002692, +0.001440] | +0.071 [-0.153, +0.306] |
| DeBERTa 401 | +0.006035 [+0.002473, +0.009584] | -0.000278 [-0.001951, +0.001340] | -0.041 [-0.275, +0.194] |
| DeBERTa 503 | +0.014507 [+0.009991, +0.019086] | +0.000594 [-0.002070, +0.003303] | -0.204 [-0.510, +0.092] |
| RoBERTa 607 | -3.936979e-6 [-1.610955e-5, +8.603104e-6] | -4.780191e-6 [-1.161354e-5, +2.238378e-6] | +0.000 [+0.000, +0.000] |
| RoBERTa 701 | +0.021743 [+0.015129, +0.028512] | -0.008737 [-0.011637, -0.005884] | +0.571 [+0.194, +0.958] |
| RoBERTa 809 | +0.008832 [+0.007382, +0.010357] | -0.000931 [-0.001377, -0.000491] | +0.153 [+0.020, +0.285] |

Intervals containing zero do not establish equivalence or noninferiority. Each model's temperature is fitted on its 512 calibration labels.
