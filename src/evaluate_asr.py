"""Evaluate base, best-scene adapter, and soft-fusion ASR with CER."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from jiwer import cer
from src.common import read_jsonl
from src.fused_inference import FusedWhisperInference, SceneRouter

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, default=Path("data/processed/aishell1/test.jsonl")); p.add_argument("--adapter-dir", type=Path, default=Path("output/adapters")); p.add_argument("--classifier", type=Path, default=Path("output/scene_classifier.pt")); p.add_argument("--limit", type=int, default=0); p.add_argument("--report", type=Path, default=Path("output/evaluation.csv")); args = p.parse_args()
    engine = FusedWhisperInference(); router = SceneRouter(args.classifier, "openai/whisper-small", engine.device) if args.classifier.exists() else None
    for directory in args.adapter_dir.iterdir() if args.adapter_dir.exists() else []:
        if (directory / "adapter_config.json").exists(): engine.load_adapter(directory.name, directory)
    rows = read_jsonl(args.manifest); rows = rows[:args.limit] if args.limit else rows; results=[]
    for row in rows:
        probs = router.predict(row["audio_path"]) if router else None
        base_prediction = engine.transcribe_base(row["audio_path"])
        fused_prediction = engine.transcribe(row["audio_path"], probs) if probs else base_prediction
        results.append({"reference":row["text"], "base_prediction":base_prediction, "fused_prediction":fused_prediction,
                        "base_cer":cer(row["text"], base_prediction), "fused_cer":cer(row["text"], fused_prediction),
                        "scene":max(probs,key=probs.get) if probs else "base", "scene_probabilities":json.dumps(probs, ensure_ascii=False) if probs else ""})
    args.report.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(results).to_csv(args.report,index=False,encoding="utf-8-sig")
    summary={"utterances":len(results), "base_cer":sum(x["base_cer"] for x in results)/max(len(results),1), "fused_cer":sum(x["fused_cer"] for x in results)/max(len(results),1), "router_used":router is not None}
    args.report.with_suffix(".summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary, ensure_ascii=False))
if __name__ == "__main__": main()
