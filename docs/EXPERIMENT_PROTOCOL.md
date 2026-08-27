# 正式实验协议与结果判定

## 1. 研究问题

主问题不是“能否让一个数字变小”，而是分离三个变量：

1. 把合成白噪/RIR换成真实退化，是否改善退化场景鲁棒性？
2. 在相同真实数据下，hard、测试时 soft 权重插值、联合微调 joint 谁更可靠？
3. 场景路由不确定时回退 base，能否降低 clean 场景伤害和危险误路由？

## 2. 固定系统

- **base**：Whisper-small，强制关闭所有 adapter。
- **hard**：路由通过阈值后只激活 argmax 专家，否则 base。
- **soft**：路由通过阈值后保留 top-2 场景、按 0.05 量化并重新归一化，再做 PEFT 权重插值；否则 base。实际权重逐条写入 `soft_applied_weights`，避免服务缓存无限组合且保证可审计。
- **joint**：以五专家融合为初始化，在五场景混合数据上联合微调后的单 adapter，不依赖测试时路由。

这四路必须使用相同文本规范化、解码配置和 test manifest。不得把旧版逐句平均 CER 与新版语料级 CER 直接横向比较。

## 3. 数据控制

- 原始 train/validation/test 使用 AISHELL-1 官方分区。
- 每个原始 utterance 可在同一 split 产生多个场景视图，但不得跨 split。
- noise/RIR 只按 asset ID 抽取，manifest 记录来源、许可证、SNR、RIR、速度率和 seed。
- test 每桶默认 1,000 条；五桶共 5,000 条。
- 真实噪声资产本身允许在 split 间重复；防泄漏约束针对语音内容 `source_id`。如果研究噪声泛化，应另做 noise-source held-out 消融，不能混入主实验。

## 4. 指标

主指标为语料级 CER：

```text
CER = (substitutions + deletions + insertions) / reference_characters
```

中文按去空格后的 Unicode 字符计数。每个 bucket 和 overall 均报告总 S/D/I、参考字符数和 CER。

比较 candidate 与 base 时按 utterance 做配对 bootstrap，默认 2,000 次，报告：

- observed candidate CER - base CER；
- 95% percentile CI；
- bootstrap 中 candidate 更好的比例。

CI 跨 0 时，不写“显著提升”或“显著退化”。

## 5. 路由指标

路由器不能只报 accuracy。至少报告：

- source-disjoint validation accuracy；
- 5×5 confusion matrix；
- temperature scaling 后 ECE；
- test 上 accepted / low_confidence / high_entropy 比例；
- 每个真实场景的回退率。

默认接受条件：最大概率 ≥0.65 且熵 ≤1.35。阈值消融必须使用 validation 确定，不能看 test 后调参。

## 6. 最小消融矩阵

| 数据 | 融合/训练 | 目的 |
|---|---|---|
| 旧合成 | 旧测试时 soft | 历史负结果，仅引用现存证据 |
| 真实退化 | base/hard/soft | 测数据与路由贡献 |
| 真实退化 | linear init + joint | 测联合梯度协同贡献 |
| 真实退化 | TIES 或 DARE init + joint | 仅当 linear 仍显示干扰时追加 |

可微 MoE/共享低秩属于 v2.1 结构性研究，不与主矩阵一起首跑。

## 7. 转正标准

“项目转正”不等于每桶都赢。最低可接受叙事是：

1. noisy/reverb/noisy_reverb 中至少有明确、可重复的改善；
2. clean 不出现不可接受的退化，或通过路由回退把退化控制住；
3. overall 差异有 CI；
4. hard/soft/joint 的差异能对应机制解释；
5. 所有结果由 `run_manifest.json` 和逐条 CSV 可追溯。

若 joint 仍无改善，结论应写成“真实退化与严格评测下联合微调未优于 base”，随后依据桶级错误分析决定是否进入 MoE，而不是继续堆方法。
