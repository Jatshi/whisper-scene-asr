"""Local-only Gradio demo for calibrated scene-adaptive ASR."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from src.common import SCENE_NAMES
from src.fused_inference import FusedWhisperInference, SceneRouter

MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-small")
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "output/v2/adapters"))
CLASSIFIER = Path(os.getenv("CLASSIFIER_PATH", "output/v2/scene_classifier.pt"))
ROUTING_MODE = os.getenv("ROUTING_MODE", "soft")
CONFIDENCE_THRESHOLD = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.65"))
ENTROPY_THRESHOLD = float(os.getenv("ROUTER_ENTROPY_THRESHOLD", "1.35"))
FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", "2"))
FUSION_WEIGHT_STEP = float(os.getenv("FUSION_WEIGHT_STEP", "0.05"))

engine = FusedWhisperInference(MODEL, fusion_top_k=FUSION_TOP_K, fusion_weight_step=FUSION_WEIGHT_STEP)
router = SceneRouter(CLASSIFIER, MODEL, engine.device) if CLASSIFIER.exists() else None
for scene in SCENE_NAMES:
    candidate = ADAPTER_DIR / scene
    if (candidate / "adapter_config.json").exists():
        engine.load_adapter(scene, candidate)


def transcribe(audio: str | None) -> tuple[str, str]:
    if not audio:
        return "", "请先提供音频。"
    if router is None or not engine.adapter_names:
        return engine.transcribe_base(audio), "路由器或场景 adapter 不完整，已安全回退 base。"
    probabilities = router.predict(audio)
    text, decision = engine.transcribe_routed(
        audio,
        probabilities,
        mode=ROUTING_MODE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        entropy_threshold=ENTROPY_THRESHOLD,
    )
    details = [
        f"route: {decision.route}",
        f"reason: {decision.reason}",
        f"confidence: {decision.confidence:.1%}",
        f"entropy: {decision.entropy:.3f}",
        "",
    ]
    details.extend(f"{name}: {probabilities[name]:.1%}" for name in SCENE_NAMES)
    if ROUTING_MODE == "soft" and decision.route != "base":
        details.extend(["", "applied: " + json.dumps(engine.soft_weights(probabilities), ensure_ascii=False)])
    return text, "\n".join(details)


with gr.Blocks(title="Whisper Scene-Adaptive Chinese ASR v2") as demo:
    gr.Markdown("# Whisper 多场景中文 ASR v2\n真实退化训练 · 校准路由 · 低置信回退")
    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="音频")
    button = gr.Button("开始识别", variant="primary")
    text = gr.Textbox(label="转写结果", lines=4)
    scene = gr.Textbox(label="路由证据", lines=10)
    button.click(transcribe, audio, [text, scene])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
