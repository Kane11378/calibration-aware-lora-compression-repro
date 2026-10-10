# Mathematical implementation

The original mathematical implementation comprises four files:

| File | Function |
| --- | --- |
| `experiments/constrained_compensation_control_01/solver.py` | Fixed-objective nonnegative compensation and certificates |
| `experiments/structured_fisher_shrinkage_01/mathematics.py` | Off-block moment reliability and S1 geometry |
| `experiments/hierarchical_fisher_shrinkage_01/shrinkage.py` | Within/cross reliability and H1 geometry |
| `experiments/global_fisher_shrinkage_ablation_01/mathematics.py` | Pooled reliability and G1 geometry |

Their source bytes and internal mathematical metadata are preserved, including
historical references to the development stages numbered 25, 27, 28 and 43.
Those references identify the inherited algebra and are not execution
instructions. `manifests/scientific_code_identity.json` records exact hashes.
NumPy, SciPy and threadpoolctl are the external imports in this closure. The static
verification command parses the files without importing them.

These files expose mathematical functions rather than a complete experiment
launcher. Model training, data loading, endpoint execution and recovery
orchestration require additional materials and assets from the author.
