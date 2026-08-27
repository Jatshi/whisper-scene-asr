# 简历与面试表述边界

## 现在可以说

- 完成 Whisper Scene ASR v2 的完整工程链路：真实退化数据、源隔离、五专家 LoRA、校准路由、低置信回退、joint 微调、四路分桶评测、配对 bootstrap 和可恢复 AutoDL 流水线。
- 在固定 AISHELL-1 派生五场景测试集（N=5,000）上，joint LoRA overall CER 为 15.05%，同次禁用 adapter 的 base 为 40.41%。
- joint-minus-base 的绝对 CER 差为 -25.36 个百分点，2,000 次 utterance-paired bootstrap 95% CI 为 [-29.39, -22.08] 个百分点。
- 路由器在 source-disjoint validation 上准确率为 88.84%，温度校准 ECE 为 1.20%；推理时 11.44% 低置信样本回退 base。
- soft routing 是被严格评测的对照，而不是被包装成成功方案：其 overall CER 19.35%，差于 hard 的 17.17%。

## 必须带上的限定词

推荐写法：

> 在 AISHELL-1 派生的五场景 source-disjoint 测试协议（N=5,000）下，构建并评测 base、hard routing、soft fusion 与 joint LoRA 四条链路；joint overall CER 15.05%，相对同次 base 下降 25.36 个百分点，配对 bootstrap 95% CI [-29.39, -22.08] 个百分点。校准路由器验证准确率 88.84%、ECE 1.20%，对 11.44% 低置信输入回退 base。

面试中应主动补一句：测试分布是一个语料派生的构造退化协议，尚需真实设备录音、多数据集和多 seed 才能外推。

## 现在不能说

- “达到或超过中文 ASR SOTA”；
- “所有真实噪声环境都能提升”；
- “soft routing 一定优于 hard routing”；
- “已完成生产部署”或“已有线上用户”；
- “结果来自分布式训练”——本次是单卡完整实验；
- “联合训练对每个场景都最好”——fast/slow 桶 soft 略优于 joint；
- 任何无法在 `evaluation.summary.json`、`router_metrics.json`、`gpu_smoke.json` 或 `run_manifest.json` 定位的数字。

## 旧版如何处理

旧版 7,176 条干净集实验可以作为“发现测试时线性融合无收益，因而推动 v2 重构”的历史背景，不要与 v2 的五场景 overall 数字直接纵向比较。两个实验的测试分布不同。

## 证据优先级

1. `output/v2/evaluation.csv`：逐条预测和编辑数；
2. `output/v2/evaluation.summary.json`：聚合与 CI；
3. `output/v2/router_metrics.json`：路由与校准；
4. `output/v2/run_manifest.json`：环境、数据和哈希；
5. `.aris/claims.json`：结论到 JSON 路径的映射。

不要把“代码实现”“smoke 通过”“完整链路跑完”“统计结果支持”混成一个模糊句子；它们是四个不同层级的证据。
