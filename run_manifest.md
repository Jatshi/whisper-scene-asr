# Execution manifest

## Local verification

- `python -m compileall -q src app.py` passed.
- `python -m pytest -q` passed: 6 tests.

## Completed GPU execution

- Platform: remote RTX 4090 (24GB VRAM).
- Data: AISHELL-1, with `120,098` train, `14,326` validation, and `7,176` test utterances.
- Augmentation: 25,000 samples, 5,000 per acoustic scene.
- LoRA: five adapters (`clean`, `fast_slow`, `noisy`, `noisy_reverb`, `reverb`), each trained for five epochs.
- Scene classifier: best validation accuracy `99.52%`.
- Full held-out evaluation: 7,176 utterances, scene router enabled.

## Evaluation result

| System | Mean CER |
| --- | ---: |
| Whisper-small base | 0.085694 |
| Scene-routed weighted LoRA fusion | 0.085963 |

The fusion result is `0.000269` CER worse than the base on this full test run. This is recorded as an observed negative result, not presented as an improvement.

## Verified deliverables

- `output/evaluation.csv` — 7,176 utterance-level results.
- `output/evaluation.summary.json` — aggregate CER summary.
- `output/scene_classifier.pt` — trained scene router.
- `whisper-scene-asr-project.tar.gz` — portable package (about 510MB), verified to contain all five adapter configurations, classifier, and evaluation data.
