# AutoDL 正式运行手册

## 1. 开机前已经固定的事项

- 正式训练只在远端 CUDA 机器运行，本地不下载 Whisper 权重、不跑训练。
- 默认数据盘需求下限 80GB；200GB 足够默认的 AISHELL-1 + MUSAN + RIR + 三 split 退化数据 + checkpoints。
- DNS Challenge 全量数据约 1TB，不会默认下载到 200GB 盘；它只能作为外接/裁剪后的额外 noise spec。
- 所有命令从项目根目录执行，模型缓存位于数据盘，不写系统盘。

## 2. 标准启动

把项目放到 `/root/autodl-tmp/whisper-scene-asr`，执行：

```bash
cd /root/autodl-tmp/whisper-scene-asr
bash scripts/run_autodl_v2.sh
```

可覆盖的主要环境变量：

```bash
PROJECT_DIR=/root/autodl-tmp/whisper-scene-asr
DATA_ROOT=/root/autodl-tmp/whisper-scene-asr-data
OUTPUT_ROOT=/root/autodl-tmp/whisper-scene-asr/output/v2
BATCH_SIZE=8
TRAIN_PER_SCENE=5000
VALIDATION_PER_SCENE=500
TEST_PER_SCENE=1000
SEED=42
```

4090 首次使用 `BATCH_SIZE=8`。如果真实 smoke 通过而训练 OOM，先降到 4；不要同时改 gradient accumulation，否则有效 batch 也变了。

## 3. 阶段与预计资源

| 阶段 | 主要工作 | GPU | 可安全重跑 |
|---|---|---:|---|
| 00 | 磁盘/CUDA 检查 | 极少 | 是 |
| 01 | 固定 Python 依赖 | 否 | 是 |
| 02 | 下载/解压/缓存 | 否 | 断点下载 |
| 02_gpu_smoke | 真模型 forward/backward | 是，数分钟内 | 是 |
| 03–05 | manifest 与退化数据 | 主要 CPU/磁盘 | marker 级 |
| 06 | 五个场景 LoRA | 是，最长 | Trainer checkpoint |
| 07 | 场景路由器 | 是 | 以 stage 为单位 |
| 08 | joint 联合微调 | 是 | Trainer checkpoint |
| 09 | 四系统完整评测 | 是，较长 | partial JSONL |
| 10–11 | 事实清单和打包 | CPU/磁盘 | 是 |

## 4. 下载速度与付费时间

流水线先做下载再做训练；日志持续写入 `output/v2/logs/pipeline.log`。若模型或 11GB MUSAN 下载长期无吞吐，应立即检查 `HF_ENDPOINT`、OpenSLR 镜像和 AutoDL 网络，不要让机器空烧。

MUSAN 官方页面提供中国镜像。当前 Python downloader 使用可恢复 `.part` 文件；中断后重跑不会从零开始。AISHELL 如果 AutoDL 公共盘已有归档，会优先使用公共盘。

## 5. 恢复与改配置

普通掉线/关 shell：重新执行同一脚本即可。SSH 断开不会因为脚本自身自动转入后台；正式启动时建议由会话管理器托管：

```bash
screen -S whisper_asr_v2
bash scripts/run_autodl_v2.sh
```

然后按 `Ctrl-a d` 脱离。

如果主动改了某阶段配置：

1. 先把旧阶段目录移动到带时间戳的备份位置；
2. 用 `reset_stage.sh` 只移除相应 marker；
3. 重跑流水线；
4. 不要手工编辑 `training_request.json` 或 `.request.json` 绕过保护。

训练 request 指纹不一致会拒绝接着旧 checkpoint 跑；评测系统/阈值变化也会拒绝复用旧 partial。

## 6. 监控

```bash
tail -f output/v2/logs/pipeline.log
nvidia-smi
du -sh /root/autodl-tmp/whisper-scene-asr-data/*
```

看训练是否健康，至少同时检查 loss、eval CER、学习率、GPU 利用率和 checkpoint 更新时间。只有显存占用、没有利用率并不代表训练在工作。

## 7. 结束条件

只有以下文件同时存在才算完整链路：

```text
gpu_smoke.json
adapters/{clean,noisy,reverb,fast_slow,noisy_reverb}/adapter_config.json
scene_classifier.pt
router_metrics.json
joint/deploy/joint/adapter_config.json
evaluation.csv
evaluation.summary.json
run_manifest.json
whisper-scene-asr-v2.tar.gz
```

任何阶段失败都不应补写数字或提前更新简历。
