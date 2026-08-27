# v2 实现与验收追踪

## 1. 边界

本轮目标不是在没有 GPU 结果时“把负结果写成正结果”，而是把一次正式实验所需的代码、数据协议、续跑机制、评测定义、证据产物和失败条件全部预先固定。状态分为：

- **本地已验证**：不需要下载模型或占用 GPU，已有自动测试证据。
- **代码已就绪，待 GPU 验证**：实现已经存在，但真实性必须由 AutoDL 真实运行证明。
- **研究门控项**：只有主路径结果满足条件后才值得投入，不应混入第一轮结论。

## 2. 规划逐项映射

| 规划要求 | 实现位置 | 自动验收物 | 当前状态 |
|---|---|---|---|
| 真实噪声/RIR | `asset_manifest.py`、`data_augmentation.py` | `assets/*.jsonl`、`dataset_report.json` | 代码就绪 |
| 保音高快慢速 | `librosa.effects.time_stretch` | manifest 的 `speed_rate` | 本地已测核心函数 |
| 来源/许可证追踪 | `asset_manifest.py` | `source`、`source_license` | 本地已实现 |
| 按源防泄漏切分 | `common.assert_disjoint_sources` | `source_disjoint: true`；冲突直接异常 | 本地单测通过 |
| 五场景 LoRA | `train_lora.py` | 5 份 `adapter_config.json` + safetensors | 待 GPU |
| 融合后联合微调 | `train_joint_lora.py` | `joint/deploy/joint/` | 待 GPU |
| Linear/TIES/DARE 初始化 | `--combination-type`、`--density` | 独立运行目录 | 代码就绪，默认仅 Linear |
| 分类器显式验证集 | `scene_classifier.py` | `router_metrics.json` | 待 GPU |
| 概率校准 | 温度缩放 + ECE | checkpoint `temperature`、report `ece` | 待 GPU |
| 低置信回退 | `routing.py` | 每条 `router_reason` | 本地单测通过 |
| base 真隔离 | `disable_adapter()` | 四路评测中的 base | 代码审计完成，待真实模型 smoke |
| hard/soft 对照 | `evaluate_asr.py` | CSV 两列预测与 CER | 待 GPU |
| joint 对照 | 同上 | joint 预测/CER | 待 GPU |
| 分桶评测 | `true_scene` + summary buckets | 5 buckets + overall | 代码就绪 |
| 标准语料级 CER | `metrics.corpus_counts` | 总 errors / 总 ref chars | 本地单测通过 |
| 配对 bootstrap 95% CI | `paired_bootstrap_difference` | low/high/P(candidate better) | 本地单测通过 |
| 批量解码 | base/hard/joint batch API | `--batch-size` | 代码就绪 |
| soft 缓存有界 | top-2/0.05 稀疏化 + `OrderedDict` + `delete_adapter` | 应用权重逐条落盘、cache 上限 | 代码审计完成，待 PEFT smoke |
| 断点续跑 | Trainer checkpoint + stage marker + request 指纹 | `training_request.json`、`.done` | 代码就绪 |
| 事实型运行清单 | `write_run_manifest.py` | 环境/指标/SHA-256 | 只会在结果齐全后生成 |
| 可移植打包 | `package_artifacts --version v2` | tar.gz | 缺任何核心产物即拒绝打包 |

## 3. 第一轮不默认执行的研究门控项

可微门控 MoE、共享 LoRA + 场景增量属于结构性新模型，不是“在旧逻辑上补工程完整性”的必要条件。它们应在以下条件同时满足后再进入 v2.1：

1. 真实数据的 base/hard/soft/joint 主实验完整跑完；
2. joint 相对测试时 soft 的差异有 CI 支撑，或出现明确的场景冲突证据；
3. 路由器在源隔离验证集上的校准质量可接受；
4. 已能区分“数据换真”与“联合训练”的贡献。

否则直接上 MoE 会同时改变数据、路由和模型结构，无法知道提升来自哪里，也会再次陷入 Demo 式堆概念。

## 4. AutoDL 开机后的硬验收顺序

1. `00_preflight`：磁盘、Python、CUDA PyTorch 和 GPU 可见。
2. `02_download`：模型、AISHELL、MUSAN、RIR 均可读。
3. `02_gpu_smoke`：真实 Whisper-small 上联合 adapter 前向/反向成功，joint 梯度有限。
4. `05_scene_corpus`：三 split 无 source overlap，五桶数量符合配置。
5. `06_scene_adapters`：每个专家都产生最佳 checkpoint 和部署 adapter。
6. `07_router`：有混淆矩阵、ECE、temperature；不能只报 accuracy。
7. `08_joint_adapter`：从融合初始化继续训练，部署包只含 joint。
8. `09_bucketed_evaluation`：四系统、五桶、overall 和 CI 全齐。
9. `10_manifest`：只从真实文件生成，不接受手填数字。
10. `11_package`：排除 checkpoints、缓存和 partial 文件。

## 5. 本地验证证据

执行：

```bash
python -m ruff check .
python -m compileall -q src app.py
python -m pytest -q
```

准备阶段记录：Ruff 通过，编译通过，21 tests passed。此记录不等于 GPU 模型链路已经通过；后者由 `gpu_smoke.json` 单独证明。
