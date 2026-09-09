# OPD v3 AutoDL 执行状态

更新时间：2026-09-10

## 最终状态

- 验证规模的三 seed 完整实验已于 2026-09-10 05:53 CST 完成。
- 最终输出：`/root/autodl-tmp/whisper-scene-asr/output/v3-opd-validation`。
- `stages/60_aggregate.done` 与 `stages/61_manifest.done` 均存在。
- `run_manifest.json` 明确记录 `status=validation_only`。
- 运行中未发现 OOM、NaN 或 Traceback。
- 完整结果已拉取到本地 F 盘，manifest 登记的 163 个文件全部通过大小与 SHA-256 校验。

## 实际运行参数

```text
SEEDS=42,7,2026
OPD_ROUNDS=1              # round 0 与 round 1，共两轮
COLLECTION_SIZE=5000
ORACLE_FRACTION=0.25
REPLAY_WINDOW=4
BATCH_SIZE=32
POLICY_BATCH_SIZE=128
POLICY_EPOCHS=10
OFFLINE_EPOCHS=140
EVAL_LIMIT=500
RUN_ABLATIONS=1
INCLUDE_JOINT=0
ALLOW_SINGLE_SEED_AGGREGATE=0
```

500 条评测不是简单取文件前 500 条，而是基于稳定哈希选择并严格分层：clean、noisy、reverb、fast_slow、noisy_reverb 各 100 条。

## 运行规模与耗时

| 项目 | 实际值 |
|---|---:|
| GPU | RTX 3080 Ti 12GB |
| 三 seed 在线采集 | 30,000 条 annotations |
| 每 seed 策略 checkpoint | init + round-0 + round-1 |
| 每 seed 评测 | 主结果、FKL、RKL、top-k 1/3、calibration、adaptive |
| Paired bootstrap | 每比较 2,000 次 |
| 总墙钟时间 | 10.20 小时 |
| 本地完整结果 | 188 文件，1,241,702,845 bytes |

Seed 42 的两轮 annotations 和 checkpoint 来自先前全量尝试，源请求保存在 `reused_seed42_source_run_request.json`；后续源代码修改只涉及分层评测限制、汇总状态和 manifest，不改变 oracle 或训练数学。其余 seed 在本次验证运行中重新完成。

## 资源基准

| ASR batch | 收集条数 | 评测条数 | smoke 墙钟时间 | 观测峰值显存 |
|---:|---:|---:|---:|---:|
| 8 | 64 | 32 | 139 秒 | 2,228MiB |
| 32 | 128 | 64 | 186 秒 | 3,896MiB |
| 64 | 128 | 64 | 178 秒 | 6,238MiB |

正式验证采用 batch=32。12GB 显存足够，显存不是瓶颈；top-k=3 多专家解码与串行消融才是主要墙钟开销。

## 过程中发现并解决的问题

1. **原全量方案耗时超过可接受预算。** 保留已完成的 Seed 42 两轮训练，改为三 seed × 两轮 × 500 条严格分层验证；最终清单强制写 `validation_only`，避免冒充完整实验。
2. **smoke 曾误跑完整测试集。** 新增 `EVAL_LIMIT` 并绑定进 `run_request.json`。
3. **截断评测可能破坏场景均衡。** 新增稳定哈希的 `stratified_limit`，保证五场景各 100 条。
4. **单 seed smoke 无法汇总。** 新增显式 `ALLOW_SINGLE_SEED_AGGREGATE`；正式验证仍要求多 seed。
5. **Whisper generation 警告。** 清理冲突的 forced decoder ids，并向全部 batch 解码路径传入 attention mask。
6. **固定门槛过度保守。** 主策略平均回退率 99.6%；使用独立 calibration split 后 hard 路由的平均回退率降至 92.8%，CER 从 33.961% 降至 32.350%。
7. **Reverse KL 策略塌缩。** 三个 seed 均触发 100% 回退，作为负结果保留而非删除。

完整指标和声明边界见 `docs/RESULTS_V3_OPD.md`。
