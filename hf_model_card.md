---
language:
- zh
pipeline_tag: automatic-speech-recognition
library_name: peft
base_model: openai/whisper-small
datasets:
- openslr/aishell
license: mit
tags:
- whisper
- lora
- chinese-asr
- acoustic-robustness
- routed-experts
---

# Whisper Scene ASR v2

This repository stores the released artifacts for [Whisper Scene ASR](https://github.com/Jatshi/whisper-scene-asr), a reproducible Chinese multi-scene ASR study built on `openai/whisper-small` and PEFT LoRA.

The v2 artifacts are under `v2/`:

- `experts/<scene>/`: final LoRA adapters for clean, noisy, reverb, fast/slow and noisy+reverb conditions.
- `joint/`: the jointly fine-tuned deployment LoRA.
- `scene_classifier.pt`: calibrated five-scene router.
- `evaluation.summary.json`, `router_metrics.json`, `gpu_smoke.json`, `run_manifest.json`: evidence and environment records.
- `whisper-scene-asr-v2.tar.gz`: portable release package excluding training checkpoints and caches.

## Evaluation snapshot

The held-out test contains 5,000 AISHELL-1-derived utterances, equally split across five constructed acoustic conditions. Corpus-level character error rate (CER):

| System | Overall CER |
|---|---:|
| Adapter-disabled Whisper-small | 40.41% |
| Hard routed expert | 17.17% |
| Soft routed fusion | 19.35% |
| Joint LoRA | **15.05%** |

Joint-minus-base absolute CER difference was -25.36 percentage points; utterance-paired bootstrap 95% CI was [-29.39, -22.08] percentage points using 2,000 samples. Router validation accuracy was 88.84%, calibrated ECE was 1.20%, and 11.44% of evaluation samples fell back to the base path.

## Important scope limitation

These numbers apply only to this single AISHELL-1-derived five-scene protocol. They do not establish state of the art, universal real-world robustness, or production readiness. Soft fusion did not outperform hard routing overall. See the GitHub [results report](https://github.com/Jatshi/whisper-scene-asr/blob/main/docs/RESULTS_V2.md) for bucket-level results and claim boundaries.

## Loading the joint adapter

```python
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(
    base,
    "jatshi/whisper-scene-asr",
    subfolder="v2/joint",
)
model.eval()
```

The router and expert paths require the project inference code because they include evidence gating, adapter selection/fusion and cache management.

## Licenses

Project code is MIT. The base model, AISHELL-1, MUSAN and OpenSLR RIRS retain their own licenses and terms. Model artifacts are derivative adapters and require the upstream Whisper-small weights at runtime.
