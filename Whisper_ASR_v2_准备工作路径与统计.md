# Whisper Scene ASR v2 最终路径、运行与交付统计

> 更新时间：2026-08-27（Asia/Shanghai）  
> 当前状态：AutoDL 12-stage 完整链路已结束；全量产物已拉取到 F 盘并完成顶层 SHA-256 对照。GitHub 与 Hugging Face 的最终发布状态以本文第 7 节为准。

## 1. 关键绝对路径

| 内容 | 绝对路径 |
|---|---|
| F 盘项目与源码 | `F:\Codex\2026-07-16\new-chat\outputs\whisper-scene-asr` |
| F 盘全量实验输出 | `F:\Codex\2026-07-16\new-chat\outputs\whisper-scene-asr\output\v2` |
| GitHub 安全发布工作副本 | `F:\Codex\2026-07-16\new-chat\outputs\whisper-scene-asr-github-delivery` |
| 待重建的纯源码包 | `F:\Codex\2026-07-16\new-chat\outputs\whisper-scene-asr-v2-source-ready.tar.gz` |
| AutoDL 项目 | `/root/autodl-tmp/whisper-scene-asr` |
| AutoDL 全量输出 | `/root/autodl-tmp/whisper-scene-asr/output/v2` |
| 原始夯实规划（未修改） | `C:\Users\jat_s\WorkBuddy\2026-06-02-09-04-45\项目夯实规划_Whisper多场景ASR.md` |

## 2. 正式运行状态

`output/v2/stages/` 中 12 个阶段全部完成：

1. `00_preflight`
2. `01_dependencies`
3. `02_assets`
4. `03_gpu_smoke`
5. `04_dataset`
6. `05_scene_adapters`
7. `06_router`
8. `07_joint_init`
9. `08_joint_adapter`
10. `09_evaluation`
11. `10_manifest`
12. `11_package`

运行环境：Python 3.10.8、PyTorch 2.3.1+cu121、Transformers 4.46.3、PEFT 0.13.2；GPU 为 NVIDIA GeForce RTX 4080 SUPER 32GB。GPU smoke 已完成真实 Whisper-small + LoRA 前向、反向与梯度有限性检查。

## 3. 数据与模型统计

| 项目 | 数量/结果 |
|---|---:|
| train | 25,000 条 |
| validation | 2,500 条 |
| test | 5,000 条 |
| 场景 | 5 个，每个 test bucket 1,000 条 |
| 专家 LoRA | 5 个 |
| Joint LoRA | 1 个 |
| train/val/test source ID | 完全隔离 |
| 路由验证准确率 | 88.840% |
| 温度校准 ECE | 1.201% |
| overall 回退率 | 11.44% |

## 4. 最终评测

| 系统 | Overall CER |
|---|---:|
| base | 40.410% |
| hard | 17.171% |
| soft | 19.353% |
| joint | **15.051%** |

joint-minus-base 为 -25.360 个百分点，2,000 次 utterance-paired bootstrap 95% CI 为 [-29.393, -22.081] 个百分点。结果只适用于本次 AISHELL-1 派生五场景协议；详细边界见 `docs/RESULTS_V2.md`。

## 5. F 盘全量产物

`output/v2` 已完整拉取：**324 个文件，1.395GB**。这不是仅含最终权重的精简包；它还保留 checkpoint、TensorBoard event、逐条评测和中间训练状态，便于恢复与审计。

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `evaluation.csv` | 2.980MB | `7b359d7c3fa462d69bafe9097537a5fec2dd6272a0523008a0e7481aa64478a1` |
| `evaluation.summary.json` | 0.013MB | `9604ecfe5941ffca9fd42e7a964941d089d1da3deb0d254462339a42ba9352ce` |
| `router_metrics.json` | 0.001MB | `a160845f7fd066dbd9314ecfd3afd6380a486f118dbc0e2e64fd0a6f4e5f4de1` |
| `scene_classifier.pt` | 1.508MB | `4f59e35233c82c6c18291b57bd77176b80f608970e5d4c916a82e2ceb1961d00` |
| `run_manifest.json` | 0.045MB | `71c4e05d8fa8c886ca3c12d2ecdc6adb4f44e485f16c010e7c908f0c04b3933b` |
| `gpu_smoke.json` | 0.001MB | `452e6ab908a0b87598a1214096b84a8eaf4c826223f1d5fdb495a44d849e9470` |
| `whisper-scene-asr-v2.tar.gz` | 347.780MB | `25434d6e6e0bc372aec3f6c0304550e07044945645e0b816379e87e962a711cd` |

以上哈希与远端对应文件一致。

## 6. 代码与文档

核心代码包括真实退化构建、五专家 LoRA、校准路由、证据门控、soft adapter LRU、joint 微调、逐条四路评测、bootstrap、12-stage 恢复和可移植打包。当前 CPU 质量基线为：

- `ruff check` 通过；
- `compileall` 通过；
- `pytest`：21 passed；
- 真实 GPU smoke：passed。

新增结果文档：

- `docs/RESULTS_V2.md`：结果、统计解释、结论边界和下一步实验；
- `docs/TROUBLESHOOTING_V2.md`：完整链路的故障、根因、修复与面试表述；
- `.aris/claims.json`：结论到原始 JSON 路径的可机读证据映射；
- `hf_model_card.md`：Hugging Face v2 模型卡源文件。

## 7. 发布状态

本节会在发布动作完成后写入最终 commit、URL 和远端文件核验。目前目标位置为：

- GitHub：`https://github.com/Jatshi/whisper-scene-asr`
- Hugging Face：`https://huggingface.co/jatshi/whisper-scene-asr/tree/main/v2`

GitHub 只发布源码、小型结果和文档；Hugging Face 发布五个专家 adapter、joint adapter、路由器、关键结果 JSON 与 347.78MB 可移植包；F 盘保留完整 1.395GB 实验目录。

## 8. 事实边界

现在可以说“完整工程链路已跑通，并在固定五场景协议上获得带置信区间的改进”。不能说“超过 SOTA”“所有真实噪声都有效”“soft 一定优于 hard”或“已经生产落地”。后续最关键的补强是多 seed、跨数据集、真实设备录音、延迟吞吐与消融。
