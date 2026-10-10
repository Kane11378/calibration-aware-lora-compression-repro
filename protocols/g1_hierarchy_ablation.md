# Pooled versus hierarchical dependency shrinkage

The G1 mechanism ablation was specified after the original source results
were known. Its estimator, comparison rules and endpoint workflow were
frozen before the additional G1 results. It complements the original
confirmation evidence.

Deployment rank fixes retained capacity. Finite calibration limits how much
empirical dependency geometry can be trusted for post-training coefficient
compensation in those retained directions.

The locked H1 geometry is `F_H = c F + (w-c) B + (1-w) D`, with
`0 <= c <= w <= 1`. G1 uses `F_G1 = g F + (1-g) D`. Its single reliability
weight pools the same disjoint within-module and cross-module moments:
`S_G = S_W + S_C`, `V_G = V_W + V_C`, and
`g = 1 - clip(V_G / S_G, 0, 1)`. When `S_G = 0`, `g = 0`.
H1 distinguishes within-module and cross-module reliability; G1 applies one
reliability to all off-diagonal dependencies. Both use the same analytic
calibration-derived shrinkage principle, without held-out outcome selection.

The frozen comparison holds the trained source, retained singular directions,
calibration rows, archived N512 source/rank ridge, coefficient constraints,
solver and raw source-distribution KL endpoint fixed. The full-pool panel is
11 sources by ranks 8/4/2, giving 33 G1 conditions. The reduced-pool panel is
the six existing BoolQ sources by N=64/128/256, the five original nested
trajectories, and ranks 8/4/2, giving 270 conditions. Together these are the
original 303 conditions. No new training or Jacobian collection was used.
All exact previously frozen subset memberships are in
`configs/finite_calibration/subset_trajectories.json`; no memberships are
constructed by the release snapshot's static verification commands.

Full-pool source effects average the three ranks equally before paired
example-level inference. The already-stored percentile 95% intervals used
10,000 PCG64 resamples with fixed seed 20261043 and shared example-index
draws across ranks. These intervals condition on each existing trained source
and fixed endpoint. Ranks and
nested trajectories are within-source perturbations, not independent source
replications; the eleven sources are not pooled as iid population draws.

The full-pool comparison gives mixed hierarchy evidence. At N512, the
H1/G1 point split is 6/5; the paired interval split is 4 favoring H1, 2
favoring G1, and 5 inconclusive. An interval crossing zero is inconclusive,
not equivalence; no equivalence margin was specified. Under reduced
calibration, H1 is lower in 71/90 equal-rank trajectory effects and 14/18
source/N means, and has lower mean normalized coefficient distance in 14/18
source/N comparisons. Distances use each method's own N512 coefficients and
the frozen retained-spectrum denominator; they are not endpoint or task scores.

At N512, G1 has lower point-estimate KL than M0, C1 and C2 in 11/11 sources,
but lower KL than S1 in only 5/11. S1 remains a strong comparator. Seed607
has an interval favoring G1 on a tiny absolute scale and is an already-known
majority-class source. Seed809 favors G1 at N512 and has mixed trajectory
signs and nonmonotonic coefficient stability under reduced calibration.
Every source, rank, trajectory and adverse or null comparison is retained.
The evidence is stronger for dependency-complexity shrinkage than for a
universally superior hierarchical granularity. No outcome-based selector was
fitted. Raw source fidelity does not establish labeled-task superiority.

`results/g1/` contains the 101 frozen g estimates, all 303 G1 raw-KL records,
archived controls, complete rank effects, coefficient stability and summaries.
`results/g1/analysis/primary_per_example/` contains example identifiers and
paired-KL difference arrays. The identifiers refer to the pinned public benchmark.
`figure_source_data/g1/` contains the exact original hierarchy SVG and CSV
inputs; the vector figure uses stored values. The original mathematical implementation is supplied under `code/`
for inspection; static verification does not import it.
