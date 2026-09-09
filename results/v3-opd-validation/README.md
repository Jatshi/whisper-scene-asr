# v3 OPD validation artifacts

This directory contains the compact, auditable subset of the completed RTX 3080 Ti experiment.

- `run_manifest.json`: environment, status, metrics, artifact sizes and SHA-256 hashes.
- `multiseed_summary.json`: three-seed aggregate for the fixed-threshold main policy.
- `run_request.json`: bound hyperparameters and source fingerprint.
- `pool_report.json`: split sizes, fingerprints and leakage check.
- `seeds/<seed>/evaluation.summary.json`: main 500-utterance stratified evaluation.
- `seeds/<seed>/ablations/`: adaptive threshold, FKL/RKL and top-k summaries.
- `seeds/<seed>/round-*/training_summary.json`: per-round policy optimization traces.

The release is deliberately marked `validation_only`: each seed uses two OPD rounds and a deterministic 500-item test subset (100 items per acoustic scene), not the full 5,000-item test. Large annotations, per-item partial files, caches and checkpoints are excluded from GitHub; the complete 1.24GB output is archived locally and model weights are released through Hugging Face.

See [`docs/RESULTS_V3_OPD.md`](../../docs/RESULTS_V3_OPD.md) for interpretation and claim boundaries.
