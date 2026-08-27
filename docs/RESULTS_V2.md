# Whisper Scene ASR v2 实验结果与结论边界

> 运行完成时间：2026-08-27（Asia/Shanghai）  
> 事实来源：`output/v2/evaluation.summary.json`、`router_metrics.json`、`gpu_smoke.json`、`run_manifest.json`  
> 审查状态：数值已通过本地 JSON 路径核验；广义结论仍标记为 `[pending Codex review]`，因为当前环境没有可调用的独立 Codex jury。

## 1. 实验到底测了什么

测试集由 AISHELL-1 原始 test split 派生，包含 clean、noisy、reverb、fast/slow、noisy+reverb 五桶，每桶 1,000 条，共 5,000 条。噪声来自 MUSAN，混响来自 OpenSLR RIRS_NOISES 中的真实 RIR；train、validation、test 按原始 `source_id` 隔离，运行清单报告 `source_disjoint=true`。

同一批样本上比较四个系统：

- `base`：禁用 PEFT adapter 的 Whisper-small。
- `hard`：选择校准路由器概率最高的场景专家；证据不足时回退 base。
- `soft`：top-2 稀疏概率加权融合专家；同样遵循回退门控。
- `joint`：以多专家融合为初始化后，在五场景混合训练集上联合微调的单个 LoRA。

CER 使用全语料 `总替换 + 总删除 + 总插入 / 总参考字符`，不是逐句 CER 的简单平均。差异区间来自以 utterance 为单位、固定 seed 42 的 2,000 次配对 bootstrap。

## 2. 主要结果

| 系统 | S | D | I | 参考字符 | CER | 对 base 的绝对差 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 23,161 | 1,080 | 5,299 | 73,100 | 40.410% | — | — |
| hard | 11,840 | 354 | 358 | 73,100 | 17.171% | -23.239 pp | [-27.320, -19.929] pp |
| soft | 12,129 | 977 | 1,041 | 73,100 | 19.353% | -21.057 pp | [-25.124, -17.439] pp |
| **joint** | **10,531** | **278** | **193** | **73,100** | **15.051%** | **-25.360 pp** | **[-29.393, -22.081] pp** |

三个候选系统的 bootstrap `probability_candidate_better` 均为 1.0。这里的含义是：在这 2,000 个重采样中，候选 CER 每次都低于 base；它不是“真实世界成功概率 100%”。

## 3. 场景分桶

| 场景 | Base | Hard | Soft | Joint | 桶内最优 |
|---|---:|---:|---:|---:|---|
| clean | 28.222% | 11.997% | 13.988% | **8.748%** | joint |
| noisy | 31.115% | 15.144% | 16.895% | **12.503%** | joint |
| reverb | 29.973% | 14.384% | 15.643% | **11.594%** | joint |
| fast/slow | 54.391% | 16.231% | **16.190%** | 16.525% | soft |
| noisy+reverb | 58.352% | 28.098% | 34.049% | **25.882%** | joint |

joint 并非每个桶都最好：fast/slow 桶里 soft 比 joint 低 0.335 pp。这说明联合训练换来了 overall 最优和更强的多数场景表现，但没有证明它对所有退化机制都占优。

## 4. 路由与拒用机制

路由器在 source-disjoint validation 上的准确率为 88.840%，温度为 0.663752，ECE 为 1.201%。推理时不是无条件使用专家：

- 4,428 / 5,000 条通过门控并使用路由结果；
- 572 / 5,000 条因低置信回退 base，overall fallback rate 为 11.44%；
- 平均最大置信度 0.8740，平均熵 0.3383；
- fast/slow 最容易判断，回退率 0.5%；clean 最难，回退率 17.5%。

混淆主要发生在 noisy、reverb、noisy+reverb 之间，符合它们共享噪声或混响属性的事实。soft 的 overall CER 高于 hard，尤其 noisy+reverb 的插入错误明显更多，因此“概率融合天然更平滑、更准确”不成立。

## 5. 运行与资源证据

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER，32,760 MiB |
| Python | 3.10.8 |
| PyTorch | 2.3.1+cu121 |
| Transformers / PEFT | 4.46.3 / 0.13.2 |
| GPU smoke | passed |
| smoke trainable params | 221,184 |
| smoke peak VRAM | 1.056 GB |
| 数据量 | train 25,000 / validation 2,500 / test 5,000 |
| 完整本地输出 | 324 files / 1.395 GB |

显存低并不等于没有用到 GPU：LoRA 只更新很少参数，而且单卡批量较小。4090/4080 级 GPU 的主要价值是计算吞吐和更大的余量，而不是必须把显存占满。

## 6. 可支持与不可支持的说法

### 可支持

- 在本项目固定的 AISHELL-1 派生五场景、5,000 条测试协议下，joint LoRA overall CER 为 15.05%，低于同次 base 的 40.41%。
- joint 相对 base 的绝对 CER 差为 -25.36 pp，配对 bootstrap 95% CI 为 [-29.39, -22.08] pp。
- 校准路由器验证准确率为 88.84%、ECE 为 1.20%，并让 11.44% 低置信样本回退 base。
- v2 已完成从真实资产下载、数据构建、训练、路由、联合微调、四路评测到发布打包的完整链路。

### 不可支持

- “真实世界所有噪声都能提升”——只测了一个语料及其构造退化。
- “达到或超过 SOTA”——没有统一协议下的外部 SOTA 对比。
- “soft routing 优于 hard routing”——本次 overall 恰好相反。
- “可直接生产部署”——尚缺真实长音频、并发、P95 延迟、漂移和故障恢复评测。
- “提升全部来自路由算法”——joint 同时改变了训练数据与优化过程，需要消融才能拆分因果贡献。

## 7. 下一轮最有价值的实验

1. 至少 3 个随机种子，报告均值、标准差和跨 seed 区间。
2. 加入真实设备录音和公开跨域中文 ASR 集，而非只依赖合成退化。
3. 做 base mixed-LoRA、oracle router、无回退、不同阈值、不同 top-k 的消融。
4. 报告 RTF、吞吐、P50/P95 延迟、峰值显存和 adapter 缓存命中率。
5. 分析 noisy+reverb 下 soft 的插入错误，验证概率融合是否破坏了 decoder 的稳定性。

## 8. 证据文件

- `output/v2/evaluation.csv`：逐条预测与编辑数。
- `output/v2/evaluation.summary.json`：本页全部 CER 与 bootstrap 数字。
- `output/v2/router_metrics.json`：路由准确率、温度、ECE、混淆矩阵。
- `output/v2/run_manifest.json`：环境、数据与制品哈希。
- `.aris/claims.json`：结论到 JSON 路径的可机读映射。
