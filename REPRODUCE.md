# Verify the archived record

Extract the version 1.0.1 archive to a clean directory and use Python 3.10 or newer:

```text
python -I -B code/verify_release_static.py
```

The command checks every listed file against `checksums.sha256` and
`manifests/file_inventory.json`. It rejects missing, changed and additional
files, symbolic links, Git metadata, unsafe paths and private payloads. It
parses all CSV rows, Python syntax and imports, JSON records and NPZ headers.
NPZ contents are authenticated without evaluating the stored numerical values.
The output is a JSON verification report. The command writes no files and
requires no network access or third-party Python package.

This verifies the archived record rather than the numerical correctness of the
experiments. The release snapshot contains complete stored tables and plotting values,
including unfavorable results. It provides the exact four-file mathematical
implementation for H1, S1, G1 and the nonnegative solver. Its external
dependencies are NumPy, SciPy and threadpoolctl. The implementation is not
imported by the static command.

The verifier also checks genuine source aliases and consistent publication
metadata. Its CFF reader supports this package's documented subset; full
CFF schema validation is a separate check. These offline checks establish
structural consistency and do not prove public availability, DOI
resolution or author authorization. See `RELEASE_METADATA.md` for the
citation and package scope.

## Experimental reproduction

Experimental reproduction requires upstream models and datasets at the
immutable revisions in `data_or_data_access/README.md`, the recorded scientific
runtime, and the original trained adapters, Jacobians and endpoint outputs.
Those assets are omitted from this release snapshot. Data-partition and subset
identities preserve the original membership; substituting a new random split
would define a different experiment. Request the omitted project assets and
execution materials from the corresponding author, subject to provider terms
and access arrangements. This archive is not a complete retraining runner.

Individual figures and their source values are supplied for inspection and
reuse. The article, Supplement and their source closures are withheld; no
manuscript or Supplement build command is provided.
