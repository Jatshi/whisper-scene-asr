# Whisper Scene ASR v3 OPD 结果

> **状态：VALIDATION ONLY。** 真实 GPU 链路已经完成，但本次为求职展示所需的验证规模实验：3 个随机种子、每个种子 2 轮在线采集与更新、每轮 5,000 条候选、500 条五场景等额分层测试子集。它证明方法和工程链路可运行，也暴露了门控退化问题；它不是完整 5,000 条测试集上的正式论文结论。

## 实验配置

| 项目 | 数值 |
|---|---:|
| GPU | RTX 3080 Ti 12GB |
| 随机种子 | 42、7、2026 |
| OPD 轮数 | 每个种子 2 轮（round 0–1） |
| 每轮采集 | 5,000 条 |
| Oracle 比例 | 25% |
| 策略训练 | 10 epochs / round，replay window=4 |
| 测试子集 | 500 条；clean/noisy/reverb/fast_slow/noisy_reverb 各 100 条 |
| Paired bootstrap | 每个比较 2,000 次 |
| 总墙钟时间 | 10.20 小时 |

OPD pool 含 27,250 条训练候选，另留 250 条 calibration；与 5,000 条 held-out test 按 `source_id` 验证无交叉。最终清单状态为 `validation_only`，`claim_eligible=false`。

## 主结果：固定门槛

同一个 500 条分层子集上的 base CER 为 33.961%。

| 系统 | Seed 42 | Seed 7 | Seed 2026 | Mean ± SD | 平均相对 base |
|---|---:|---:|---:|---:|---:|
| OPD hard | 33.671% | 33.961% | 33.823% | 33.818% ± 0.146% | -0.143 pp |
| OPD soft | 33.657% | 33.961% | 33.823% | 33.814% ± 0.153% | -0.148 pp |
| Oracle | 11.764% | 11.889% | 11.861% | 11.838% ± 0.065% | -22.124 pp |

固定门槛的平均回退率为 **99.6% ± 0.4%**。Seed 7 完全回退到 base；Seed 42 与 Seed 2026 也只让极少数样本进入专家动作。因此主结果只能说明固定门槛没有造成明显灾难，不能说明 online OPD 已经学成有效的专家路由器。

## 关键结果：独立校准门槛

每个 seed 只在独立的 250 条 calibration split 上网格搜索门槛，三个 seed 均选到 `confidence=0.45`、`entropy=1.5`，随后只在 500 条测试子集上评测一次。

| 系统 | Seed 42 | Seed 7 | Seed 2026 | Mean ± SD | 平均相对 base |
|---|---:|---:|---:|---:|---:|
| Adaptive hard | 31.454% | 32.839% | 32.756% | **32.350% ± 0.777%** | **-1.612 pp** |
| Adaptive soft | 35.583% | 39.601% | 32.839% | 36.008% ± 3.401% | +2.046 pp |

Adaptive hard 在三个 seed 上相对同子集 base 的 paired-bootstrap 胜出概率均为 **1.000**，平均回退率从 99.6% 降至 **92.8% ± 3.0%**。这支持“校准门槛能让 hard OPD 路由在本验证子集上稳定获益”。Adaptive soft 方差很大且平均退化，说明当前动态权重融合仍不稳定。

## 消融结果

| 变体 | Hard CER（3-seed mean ± SD） | Soft CER（3-seed mean ± SD） | 平均回退率 | 观察 |
|---|---:|---:|---:|---|
| Online OPD，固定门槛 | 33.818% ± 0.146% | 33.814% ± 0.153% | 99.6% | 门槛过严，几乎总回退 |
| Online OPD，校准门槛 | **32.350% ± 0.777%** | 36.008% ± 3.401% | 92.8% | hard 稳定改善，soft 不稳定 |
| Offline FKL | 32.936% ± 0.618% | 34.266% ± 2.750% | 95.5% | hard 三 seed 均优于 base |
| Offline RKL | 33.961% ± 0.000% | 33.961% ± 0.000% | 100.0% | 策略塌缩为全回退 |
| Top-k=1 | 33.818% ± 0.146% | 33.809% ± 0.152% | 99.6% | 与固定门槛主结果接近 |
| Top-k=3 | 33.818% ± 0.146% | 34.770% ± 1.509% | 99.6% | soft 融合扩大后反而变差 |

最有价值的失败案例是 reverse KL：三个 seed 都变成 100% uncertainty fallback。它说明 reverse KL 的 mode-seeking 特性叠加当前安全门控会放大保守动作，而不是自动带来更确定的路由。

## 与 v2 的关系

v2 在完整 5,000 条测试集上报告 hard 17.171%、soft 19.353%、joint 15.051%。v3 本次只评测确定性分层抽取的 500 条子集，而且 OPD 策略将 base 作为安全动作、绝大多数样本触发回退。因此：

- 可以在 v3 内部比较同一 500 条上的 base、OPD、Oracle 和各消融；
- 不应把 v3 的 32.350% 与 v2 的 17.171% 当作严格同协议排行榜；
- 即使只看数值，当前 v3 也没有超过 v2 routed expert，原预注册验收门槛未达到；
- 这次实验验证的是 OPD 后训练与评测基础设施，不是“v3 性能全面升级”。

## 结果支持与不支持的声明

本次结果支持：

- 完成真实 CER oracle、两轮在线收集/更新、滑窗 replay、三 seed、分层评测、门槛校准和消融的端到端工程链路；
- 在 500 条验证规模子集上，独立校准后的 hard OPD 相对同次 base 平均降低 1.612 个 CER 百分点，三个 seed 的 paired-bootstrap 胜出概率均为 1.000；
- 固定门槛、reverse KL 和 soft top-k 扩展存在可复现的退化模式。

本次结果不支持：

- OPD 已超过 v2 hard/soft/joint；
- 500 条子集结果可替代完整 5,000 条正式测试；
- soft routing 总是优于 hard routing；
- AISHELL-1 人工退化结果能外推到任意真实设备、真实噪声或生产环境。

## 可审计产物

GitHub 的 `results/v3-opd-validation/` 保存小型汇总、每个 seed 的主结果、全部消融摘要、阈值与训练摘要。Hugging Face 的 `v3-opd-validation/` 保存最终/逐轮策略权重及同一组证据文件。1.24GB 的完整原始输出保存在本地 `output/v3-opd-validation/`，包含六份 5,000 条 annotations、逐条 CSV、partial JSONL、日志和 GPU telemetry。

机器可读入口：

- `results/v3-opd-validation/run_manifest.json`
- `results/v3-opd-validation/multiseed_summary.json`
- `results/v3-opd-validation/run_request.json`
- `results/v3-opd-validation/pool_report.json`
