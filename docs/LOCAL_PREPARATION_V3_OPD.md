# OPD v3 本地准备统计

## 结论

截至 2026-09-09，OPD 后训练的本地实现已完成，可以进入 AutoDL 的“先小规模 smoke、再正式三 seed”阶段。本轮没有连接远端、没有启动 GPU、没有生成或宣称任何 v3 实验成绩。

## 工作路径

- 规范 Git 仓库：`F:\Codex\2026-07-16\new-chat\outputs\whisper-scene-asr-github-delivery`
- 本地分支：`feature/opd-v3`
- 含 v2 完整 output 的工作副本：`F:\Codex\2026-07-16\new-chat\outputs\whisper-scene-asr`
- 实现依据：`C:\Users\jat_s\WorkBuddy\2026-06-02-09-04-45\OPD项目01_Whisper路由在线策略蒸馏_实现规划.md`
- 已生成的本地源码包：`F:\Codex\2026-07-16\new-chat\outputs\whisper-scene-asr-opd-v3-local-ready.tar.gz`（同目录有 `.sha256` 校验文件）

## 新增与修改

核心代码：`opd_policy.py`、`opd_trainer.py`、`oracle_router.py`、`evaluate_opd.py`；辅助代码：训练池防泄漏、validation 阈值校准、三 seed 聚合、完成态 run manifest；推理层增加 direct soft weights 与 encoder summary batch 接口。

运行层：数据丢失时可重建但不重训 v2 专家；主脚本覆盖三 seed、困难样本 on-policy 采样、25% 分层 oracle、四轮 replay、最终 oracle/test 评测；消融覆盖 offline KD、FKL/RKL、top-k、固定/校准阈值，并提供完整 on-policy RKL 与可选 joint-action 入口。

文档层：方法与公式、AutoDL 手册、结果模板和 README 状态边界。结果模板明确为 `NOT RUN`，避免把目标值或 v2 对照误写成 OPD 结果。

## 本地验证证据

```text
Ruff check: passed
Ruff format --check: passed
compileall: passed
pytest: 37 passed
bash -n: prepare_opd_data.sh / run_opd_v2.sh / run_opd_ablations.sh passed
CLI parse smoke: opd_trainer / oracle_router / build_opd_pool /
                 aggregate_opd / calibrate_opd_thresholds / write_opd_manifest passed
secret scan: no SSH endpoint or password introduced into code/docs/tests
```

测试包含先失败再实现的四个规划契约：loss、oracle annotation、动作顺序、soft 权重；另有 annotation→policy checkpoint 的 CPU 端到端 smoke。GPU Whisper/PEFT 解码、真实数据规模与性能指标必须在 AutoDL 才能验证。

## 开机后的执行顺序

1. 上传源码包并核对 SHA-256；不覆盖远端已有 `output/v2`。
2. 用 `COLLECTION_SIZE=128, EVAL_LIMIT=64, SEEDS=42, OPD_ROUNDS=0, RUN_ABLATIONS=0` 跑独立 smoke 输出目录；正式实验保持 `EVAL_LIMIT=0`，完整评测 5,000 条测试样本。
3. 检查 annotation support、策略 checkpoint、test 小链路、显存与磁盘增长。
4. smoke 通过后，用默认参数正式执行 `scripts/run_opd_v2.sh`。
5. 只有 `output/v3-opd/run_manifest.json` 出现且 `status=complete`，才拉回本地并根据真实结果更新 `RESULTS_V3_OPD.md`。

## 尚未发生的事情

- 没有 AutoDL GPU 运行；
- 没有 OPD CER、oracle gap、三 seed 方差或消融结论；
- 没有 commit、push GitHub 或上传 Hugging Face；
- 没有修改、删除 v2 远端/本地产物。
