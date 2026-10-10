# Beyond Rank reproducibility materials

Version 1.0.1. These materials accompany *Beyond Rank: Calibration-Aware Dependency Control for Post-Training LoRA Compression* by Kang Wang.

Deployment rank limits retained capacity, while finite calibration limits the
dependencies that can be estimated reliably. The archived experiments examine
calibration-derived shrinkage for source-preserving LoRA compression. The
comparison with pooled G1 shrinkage gives mixed evidence for the additional
within-module and cross-module split in H1.

## Contents

- `results/` contains complete stored results, including adverse and inconclusive comparisons.
- `tables/` contains CSV exports of the result tables.
- `figure_source_data/` contains stored plotting values and provenance; `figures/` contains individual vector figures.
- `protocols/` describes the method and experimental designs.
- `code/experiments/` contains the original H1, S1, G1 and constrained-solver implementation.
- `configs/`, `data_or_data_access/` and `environment/` describe settings, data partitions, provider revisions and the recorded runtime.
- `manifests/` and `checksums.sha256` identify the exact files and their licensing scope.

The article and Supplement, their LaTeX sources, submission correspondence,
Highlights and graphical abstract are held separately by the author. This
release snapshot contains no full preprint and cannot rebuild those documents.

## Verification and reproduction

With Python 3.10 or newer, run from a clean extraction:

```text
python -I -B code/verify_release_static.py
```

The standard-library verifier checks exact membership, hashes, CSV structure,
Python syntax and local imports, NPZ container schemas, licensing coverage and
content hygiene. It does not import the scientific implementation or recompute
results. See `REPRODUCE.md` for its scope and the requirements for experimental
reproduction.

The eleven NPZ files hold example identifiers and derived paired KL values.
Identifiers can be matched to the pinned public benchmark; they are not
anonymized participant data. No benchmark text, individual task labels, logits,
Jacobians, trained weights or checkpoints are distributed. These omitted assets
and the original scientific runtime are needed to repeat the experiments.

The full-pool H1/G1 point directions are 6/5; paired intervals favor H1 for four
sources, G1 for two and are inconclusive for five. Reduced-calibration H1 has
lower KL than G1 in 71/90 related trajectory comparisons and than S1 in 20/90.
Ranks and nested trajectories are within-source perturbations. The archived
nonmonotonic paths, majority-class source607, source809 and PARA/Spectral
counterexamples remain part of the evidence. Source fidelity and labeled task
performance are distinct outcomes.


The data_aliases.json manifest maps genuine alternative figure-export paths
to their byte-identical targets. Canonical paths can be used directly.

## Citation and licensing

Cite: Kang Wang. *Beyond Rank: Calibration-Aware Dependency Control for Post-Training LoRA Compression*. Version 1.0.1. Software and reproducibility
materials. CITATION.cff supplies the machine-readable citation.

Kang Wang is the sole author. Contact: Kane.97022@gmail.com;
ORCID: https://orcid.org/0009-0006-7727-1214.

Project-authored Python code is MIT licensed. Project-authored results, data
and documentation are CC BY 4.0 under the per-file allocation in LICENSE
and manifests/license_scope.json. Third-party rights and exceptions remain
in force. The formal AI-use declaration is provided in AI_USE.md.
