# H1 method specification

H1 uses the original implementation identified by commit
`8f75279dc855fa40f1641f76b4322ca85fda0599` and core Git blob
`a368f5a8234ccc213b76d5d7a6a0b45c1f43a131`. Exact file hashes are in
`manifests/scientific_code_identity.json`.

## Representation and geometry

Each trained rank-32 LoRA update is represented by its effective SVD, including
LoRA scaling. Compression retains the first k spectral directions in every
adapted linear module, with k in {8,4,2}. The DeBERTa setting has 72 such linear
modules. A module is an adapted transformation rather than a Transformer block.

For N=512 source-calibration inputs, define

```text
j_i = grad_sigma(logit_1 - logit_0)
p_i = sigmoid(logit_1 - logit_0)
z_i = sqrt(p_i * (1-p_i)) * j_i
F = Z.T @ Z / N
```

F is the uncentered binary model-Fisher/GGN second moment. Coefficient
construction uses no calibration labels. Source training, stratification and
optional temperature calibration use labels.

B retains the within-module 32-by-32 diagonal blocks of F, and D=diag(F).
For each disjoint off-diagonal group, within-module and cross-module, use

```text
v_ab = sum_i (z_ia*z_ib - F_ab)^2 / (N*(N-1))
eta_group_raw = 1 - clip(sum_group v_ab / sum_group F_ab^2, 0, 1)
w = eta_within_raw
c = min(eta_cross_raw, w)
F_H = c*F + (w-c)*B + (1-w)*D
```

The implementation specifies zero-signal and floating-point handling.
Weights are estimated once from all source coordinates, independently of
target rank. They are analytic shrinkage intensities, not probabilities that
interactions are real. The nesting 0<=c<=w<=1 is a construction choice that
ensures positive semidefiniteness; no universal estimator-optimality result is
claimed.

For retained coordinates S and removed coordinates C, set delta_C=-sigma_C
and solve

```text
min_x 0.5*delta.T@F_H@delta + 0.5*d*||x||^2
delta_S = x, sigma_S + x >= 0
d = 1e-3*max(mean(diag(F_SS)), 1e-12)
```

The geometry is used consistently for retained-retained and retained-removed
terms. The original nonnegative solver and its KKT/duality certificates are
used directly. Genuine width-k factors have alpha=16; trained heads are copied
unchanged. Parameters are FP32, inference BF16, derivatives FP32 without
autocast, and small algebra float64. In the calibration-size intervention,
the archived N512 source/rank ridge is held fixed at reduced sizes.

## Properties and evidence scope

F_H is a convex combination of positive-semidefinite matrices and has the
same diagonal as F. Positive ridge makes the empirical retained-coordinate
problem strictly convex with a unique constrained minimizer. At (w,c)=(0,0)
the solution recovers spectral M0; (1,0), (1,1) and (1,c) recover C1, C2 and
the S1 family, respectively. These properties concern the empirical objective.
Finite rank deletion, rounding and estimation error limit the local quadratic
approximation to actual-network KL.

M0 is the primary practical comparator; C1, S1 and C2 are mechanistic controls.
Raw source KL, labeled cross-entropy, accuracy and temperature-calibrated
cross-entropy are reported separately. Every method receives the same optional
calibration procedure; raw KL references the original uncalibrated source.

SST-2 and QNLI informed H1 development. DeBERTa BoolQ is fixed-method new-task
confirmation, and RoBERTa BoolQ is second-backbone replication on the same raw
IDs. Ranks and nested calibration trajectories do not add independent source
replications. Absolute effects accompany relative changes, including the
majority-class source607. Lower source KL does not imply improved task loss or
accuracy.
