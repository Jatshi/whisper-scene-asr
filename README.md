# Whisper Scene ASR

<p align="center">
  <strong>Scene-aware Chinese automatic speech recognition with routed LoRA adapters.</strong><br />
  A reproducible Whisper-small study that treats acoustic condition as a first-class inference signal.
</p>

<p align="center">
  <a href="https://huggingface.co/jatshi/whisper-scene-asr"><img src="https://img.shields.io/badge/🤗%20Model-Whisper%20Scene%20ASR-ffcc4d?style=for-the-badge" alt="Hugging Face model" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20PEFT-ee4c2c?style=for-the-badge" alt="Framework" />
  <img src="https://img.shields.io/badge/License-MIT-1f6feb?style=for-the-badge" alt="MIT license" />
</p>

![Whisper Scene ASR interactive demo](assets/readme/sounddet-demo.gif)

> The animation mirrors the shipped local demo flow: audio input → acoustic-scene routing → weighted LoRA fusion → transcript. It uses the project’s real evaluation configuration and does not claim an improvement beyond the reported results.

## Why this project

Speech recognizers trained on broadly mixed speech can behave differently under clean, noisy, reverberant, and tempo-shifted audio. Whisper Scene ASR keeps one frozen Whisper-small base and learns a small LoRA adapter per acoustic condition. A scene classifier predicts routing weights at inference time, and PEFT combines adapter weights before decoding.

This repository is an experiment package, not a claim that routed LoRA universally improves Whisper. On the recorded full held-out run, fusion was slightly worse than the base model. The negative result is intentionally retained for reproducibility.

## Experiment snapshot

| Item | Value |
| --- | ---: |
| Base recognizer | `openai/whisper-small` |
| Dataset | AISHELL-1 Chinese speech |
| Scene adapters | clean, fast/slow, noisy, noisy+reverb, reverb |
| Augmented samples | 25,000, 5,000 per scene |
| Scene-classifier validation accuracy | 99.52% |
| Held-out utterances | 7,176 |
| Base CER | 0.085694 |
| Routed weighted-LoRA CER | 0.085963 |

The observed fusion delta is **+0.000269 CER**. Treat it as a negative result and a starting point for routing and adapter-ablation work.

## System design

```mermaid
flowchart LR
    A[Audio file] --> B[Whisper encoder]
    A --> C[Scene classifier]
    C -->|p(clean/noisy/reverb/tempo)| D[Adapter weights]
    D --> E[PEFT weighted adapter fusion]
    B --> F[Whisper decoder]
    E --> F
    F --> G[Chinese transcript]
```

## Feature matrix

| Capability | Included |
| --- | :---: |
| AISHELL manifest construction | ✓ |
| Deterministic acoustic augmentation | ✓ |
| Per-scene LoRA training | ✓ |
| Frozen-encoder scene classifier | ✓ |
| Probability-weighted adapter routing | ✓ |
| Utterance-level CER export | ✓ |
| Local Gradio demo | ✓ |
| Large model package on Hugging Face | ✓ |

## Quick start

```bash
git clone https://github.com/Jatshi/whisper-scene-asr.git
cd whisper-scene-asr
python -m venv .venv
.venv/Scripts/activate  # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Download the experiment artifacts from [Hugging Face](https://huggingface.co/jatshi/whisper-scene-asr) and extract them into `output/`. The upstream Whisper-small base is downloaded by Transformers on first use.

```bash
python app.py
# Open http://127.0.0.1:7860
```

For a full GPU reproduction, place AISHELL-1 where the data scripts can access it and run:

```bash
bash scripts/run_gpu.sh
```

Run a local code check before a long job:

```bash
python -m compileall -q src app.py
python -m pytest -q
```

## Model files

Large files are deliberately not committed to Git. They live in [jatshi/whisper-scene-asr](https://huggingface.co/jatshi/whisper-scene-asr).

| Artifact | Purpose |
| --- | --- |
| `whisper-scene-asr-project.tar.gz` | Portable experiment archive: five adapters, router and CER outputs. |
| `scene_classifier.pt` | Frozen-Whisper-encoder acoustic scene router. |
| `evaluation.csv` | 7,176 utterance-level CER rows. |

## Edge AI & inference notes

The routing logic is intentionally light: the classifier runs on frozen encoder features, then adapter probabilities determine a weighted fusion. The expensive component is still Whisper-small. For edge deployment, export or quantize the base recognizer separately, keep the classifier and LoRA adapters on-device, and cache the selected/fused adapter for repeated acoustic environments.

## Repository layout

```text
├── app.py                     # Local Gradio inference demo
├── assets/readme/             # README demo animation
├── results/                   # Small, versioned aggregate results
├── scripts/                   # Download and GPU-run entry points
├── src/                       # Data, augmentation, training, routing, evaluation
└── tests/                     # Unit tests
```

## Reproducibility and git policy

- Git contains source, tests, small result summaries and documentation.
- Hugging Face stores weights, adapters, archives and other large artifacts.
- `output/`, model caches, datasets and credentials are ignored.
- Do not interpret the routed score as an improvement claim without new ablations and confidence intervals.

## Citation

If you build on this implementation, cite the upstream [Whisper](https://github.com/openai/whisper) and [PEFT](https://github.com/huggingface/peft) projects, plus the AISHELL-1 dataset. A project-specific paper citation will be added when available.

## License

MIT. The code is released under MIT; the upstream model and dataset retain their own terms.
