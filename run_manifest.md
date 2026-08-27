# Whisper Scene ASR v2 execution manifest

The machine-readable source of truth is `output/v2/run_manifest.json`; a small copy is versioned at `results/run_manifest.json`.

## Completed execution

- Status: `complete`, 12/12 pipeline stages finished.
- Platform: Linux, Python 3.10.8, PyTorch 2.3.1+cu121.
- GPU: NVIDIA GeForce RTX 4080 SUPER, 32,760 MiB.
- Base model: `openai/whisper-small`.
- Data: AISHELL-1-derived five-scene protocol; train 25,000, validation 2,500, test 5,000.
- External degradations: MUSAN noise and OpenSLR RIRS_NOISES real RIR.
- Split integrity: `source_disjoint=true`.
- Models: five scene LoRA experts, calibrated scene router and one jointly fine-tuned LoRA.

## Evaluation result

| System | Overall corpus CER |
|---|---:|
| Adapter-disabled base | 0.404104 |
| Hard routed expert | 0.171710 |
| Soft routed fusion | 0.193529 |
| Joint LoRA | **0.150506** |

Joint-minus-base was -0.253598 absolute CER. The utterance-paired 2,000-sample bootstrap 95% interval was [-0.293931, -0.220806]. This result is scoped to the fixed AISHELL-1-derived evaluation and is not a universal robustness or SOTA claim.

## Router

- Validation accuracy: 0.888400.
- Temperature-scaled ECE: 0.012015.
- Overall fallback rate: 0.1144.
- Encoder passes per utterance: 1.

## Verified deliverables

- `output/v2/evaluation.csv`: 5,000 utterance-level rows for base/hard/soft/joint.
- `output/v2/evaluation.summary.json`: aggregate and five-bucket metrics plus paired bootstrap.
- `output/v2/scene_classifier.pt`: calibrated router.
- `output/v2/joint/deploy/joint/`: deployment joint adapter.
- `output/v2/run_manifest.json`: environment and artifact SHA-256 ledger.
- `output/v2/whisper-scene-asr-v2.tar.gz`: 347.78MB portable release package.

The complete F-drive copy contains 324 files and 1.395GB. Key top-level hashes are documented in `Whisper_ASR_v2_准备工作路径与统计.md`.
