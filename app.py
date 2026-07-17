import os
from pathlib import Path
import gradio as gr
from src.fused_inference import FusedWhisperInference, SceneRouter

MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-small"); ADAPTER_DIR=Path(os.getenv("ADAPTER_DIR", "output/adapters")); CLASSIFIER=Path(os.getenv("CLASSIFIER_PATH", "output/scene_classifier.pt"))
engine=FusedWhisperInference(MODEL); router=SceneRouter(CLASSIFIER, MODEL, engine.device) if CLASSIFIER.exists() else None
if ADAPTER_DIR.exists():
    for item in ADAPTER_DIR.iterdir():
        if (item / "adapter_config.json").exists(): engine.load_adapter(item.name, item)
def transcribe(audio: str | None):
    if not audio: return "", "Please provide an audio file."
    probs=router.predict(audio) if router else ({name:1/len(engine.adapter_names) for name in engine.adapter_names} if engine.adapter_names else None); text=engine.transcribe(audio, probs)
    info="Scene classifier unavailable; using uniform adapter fusion." if router is None and probs else ("No adapter loaded; using base model." if probs is None else "\n".join(f"{name}: {value:.1%}" for name,value in sorted(probs.items(), key=lambda item:-item[1])))
    return text, info
with gr.Blocks(title="Whisper Scene-Adaptive Chinese ASR") as demo:
    gr.Markdown("# Whisper 场景自适应中文 ASR")
    audio=gr.Audio(sources=["upload","microphone"], type="filepath", label="音频"); button=gr.Button("开始识别",variant="primary"); text=gr.Textbox(label="转写结果",lines=4); scene=gr.Textbox(label="声学场景概率",lines=6); button.click(transcribe,audio,[text,scene])
if __name__ == "__main__": demo.launch(server_name="127.0.0.1",server_port=7860)
