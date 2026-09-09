# OPD v3 方法与实现约定

## 目标

v2 的路由器学习“这是哪种声学场景”，但场景标签不等于最低 CER 的动作。v3 保留 Whisper-small、五个 LoRA 专家和 v2 的评测口径，只后训练一个小策略，使它直接学习在每条语音上选择 `base / 五专家 / soft` 中的低 CER 路径。默认动作空间不含 joint；joint 只作为 15.051% CER 的参照和可选消融。

## 精确定义

这是单步 contextual bandit，不是多步 trajectory RL。状态是冻结 Whisper encoder 隐状态沿时间维的 mean+std（1536 维）；动作顺序被版本化为：

`base, clean, noisy, reverb, fast_slow, noisy_reverb, soft`

策略为 `1536 → 256 → GELU → dropout(0.1) → 7`。策略温度控制动作分布；fusion 温度只把五个专家 logits 变成 top-k soft 权重，绝不让 base/soft 的 meta-action logits混入 adapter 权重。

奖励为 `-CER`，advantage 为 `CER_base - CER_action`。每条采集样本至少解码 base 和当前策略采样动作；按 scene 分层抽取 25% 样本补齐全部动作，生成 CER oracle 和 `softmax(-CER/τ)` 教师。部分标注的 KL 只在已观察动作 support 上归一化，不能把“未解码”误当作零概率；若非 oracle 样本恰好只观察到 base，一个动作没有相对偏好信息，因此该行 KL 明确置零，防止静默塌缩到 base。

总损失为：

`L = λKD·KL(q_teacher || π) + λPG·[-A log π(a|s)] - λH·H(π)`

注意熵项是负号，因为训练器最小化 loss；若写成 `+λH·H` 会鼓励低熵塌缩。reverse-KL 是消融 flag。困难度权重为 `1/(1+p_old(a|s))`；按置信/熵阈值判定的 fallback 样本仅在 KD/PG 监督项上乘 0.5，避免在天然歧义样本上过拟合，同时不削弱全局防塌缩熵。每轮采集后只保留最近四轮 replay，并依据采集分布平均熵小步调整策略温度。

## soft 动作与缓存正确性

soft 是动态动作：同一音频在不同策略 checkpoint 下可能有不同专家权重。因此 oracle request 指纹同时绑定：manifest fingerprint、policy SHA-256、动作顺序、top-k、fusion 温度、量化步长、seed 与 oracle 比例。任一项变化都会拒绝在原目录续跑，防止标签污染。

相同量化权重的语音会分组 batch 解码；融合 adapter 使用原有有界 LRU 缓存。直接权重接口只接受已加载 adapter 的正权重，并再次归一化。

## 数据边界

OPD pool 只由 v2 train 和 90% validation 构成；validation 中按稳定 source hash 留出的 10% nested calibration 不进入策略训练，只用于自适应阈值消融。这是对原规划“train+validation 全进 pool”的必要防过拟合修正。`build_opd_pool.py` 检查 pool、calibration、test 的 clean-source identity 两两不交叉。test 不参与采样、训练、温度调整或阈值选择。

## 输出与验收

三个 seed 为 42、7、2026。每个 seed 产生逐轮 annotation、策略 checkpoint、逐条 test CSV、分桶 corpus CER、2,000 次 paired bootstrap 与 oracle gap。`write_opd_manifest.py` 只有在所有 seed 汇总存在时才写 `status=complete`，并记录文件哈希。

主门槛：OPD hard CER ≤ 17.171%，且相对 v2 hard 的 paired bootstrap `probability_candidate_better ≥ 0.95`；OPD soft CER ≤ 19.353%。未运行 AutoDL 前，仓库只能声称“实现和本地测试完成”，不能声称达到这些门槛。

## 实现相对规划稿的三处必要校正

1. 单次路由没有状态转移，严格称为 contextual-bandit on-policy distillation，不宣传为多步 RL。
2. entropy 正则在最小化式中使用负号，修复规划文字中会导致塌缩的符号。
3. dynamic soft oracle 强制绑定 policy hash；否则换策略后复用 soft CER 会产生静默错误。
