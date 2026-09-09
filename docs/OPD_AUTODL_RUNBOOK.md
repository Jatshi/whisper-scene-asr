# OPD v3 AutoDL 运行手册

## 当前状态

代码、测试、数据重建入口、多 seed 主流程、消融、续跑标记和结果审计已经在本地准备；GPU 实验尚未运行。以下命令留到用户明确通知连接新的 AutoDL 实例后执行。

## 远端目录和前置条件

- 项目：`/root/autodl-tmp/whisper-scene-asr`
- v2 产物：`output/v2/scene_classifier.pt`、五专家 adapter、`evaluation.csv`
- 数据：默认 `/root/autodl-tmp/whisper-scene-asr-data`
- OPD 输出：`output/v3-opd`

若之前为节省磁盘删除了 raw/scenes 数据，`prepare_opd_data.sh` 会按 v2 的相同 seed 和规模重建；它不会重训已有五专家。开始前建议至少保留 80 GB 数据盘余量。

## 正式运行

```bash
cd /root/autodl-tmp/whisper-scene-asr
screen -S whisper_opd_v3
bash scripts/run_opd_v2.sh
```

默认执行 3 seed、5 次 on-policy 采集/更新（round 0–4），每轮按当前策略困难度从全池无放回抽 5,000 条，replay window 4、oracle 分层比例 25%，并运行 update 数量近似匹配的 offline-KD、reverse-KL offline、top-k 1/3 和 nested-calibration 阈值校准。1536 维 encoder summary 只构建一次缓存，后续策略变化只重算轻量 MLP 概率。原 validation 的 10% 按稳定 source hash 留作 calibration，不进入 OPD pool/test。完整 on-policy reverse-KL 与 joint-action 探索性消融：

```bash
bash scripts/run_opd_ablations.sh
RUN_JOINT_ACTION=1 bash scripts/run_opd_ablations.sh
```

joint-action 不是主结果，不应与“不加载 joint 逼近 joint”的主张混写。

## 成本控制

先用以下 smoke 覆盖 1 seed、1 round 和少量数据副本；正式 run 不使用 `--limit` 训练，以免改变目标分布。

```bash
SEEDS=42 OPD_ROUNDS=0 RUN_ABLATIONS=0 COLLECTION_SIZE=128 EVAL_LIMIT=64 ORACLE_FRACTION=0.05 \
  OUTPUT_ROOT=/root/autodl-tmp/whisper-scene-asr/output/v3-opd-smoke bash scripts/run_opd_v2.sh
```

`EVAL_LIMIT` 只限制 smoke 的实际解码评测条数，并被写入 `run_request.json`；默认值为 `0`，表示正式实验评测完整的 5,000 条 held-out test，不能在正式结果中设置非零值。

采集比策略 MLP 训练昂贵得多。每条至少执行 base+1 个动作，25% 子集执行完整动作；soft 会按量化权重分组 batch。若显存不足，降低 `BATCH_SIZE`，不要改变 `POLICY_BATCH_SIZE` 来解决 ASR OOM。

## 续跑与故障处理

- 成功 stage 写入 `output/v3-opd/stages/*.done`；失败 stage 不写 marker，重跑同一命令即可。`run_request.json` 绑定关键超参和 OPD 源码哈希；任一变化都会在读取 marker 前拒绝混跑。
- oracle/evaluation 各有 `.request.json`。参数或 policy hash 变化时会主动拒绝旧 partial；创建新的 `OUTPUT_ROOT`，不要删除审计证据后强续。
- 日志为 `output/v3-opd/logs/pipeline.log`。
- `annotations.jsonl` 逐 batch append；CSV 评测先写 `.partial.jsonl`，完成后排序导出。
- 出现 OOM：确认具体 stage，减半 ASR `BATCH_SIZE` 后在新输出目录运行。出现 NaN：保留日志、request 和 checkpoint，不要把该 seed 纳入汇总。

## 完成判据

只有 `output/v3-opd/run_manifest.json` 存在且 `status=complete` 才代表主链完成。随后检查：

1. `multiseed_summary.json` 有三个 seed；
2. 每个 seed 的 `evaluation.summary.json` 有 overall 和五场景桶；
3. `paired_vs_v2.opd_hard_vs_v2_hard` 存在；
4. oracle gap、fallback rate、策略哈希可追溯；
5. 再依据真实输出填写 `RESULTS_V3_OPD.md`，不得手填预计数值。
