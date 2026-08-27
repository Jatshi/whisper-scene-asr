# Whisper Scene ASR v2：完整链路踩坑、根因与修复

这份文档不把过程包装成“一次成功”。它记录真正影响完整实验的故障、为什么会发生、如何定位，以及下一次怎样更快处理。面试时可以按“现象—假设—证据—修复—防复发”来讲。

## 1. 下载完整 MUSAN 过慢，GPU 等数据空烧

**现象**：原计划拉取完整 MUSAN，但云端链路慢且不稳定，下载时间远大于训练本身。

**根因**：实验只需要噪声类素材，却按全量语料思路下载；同时 Hugging Face 官方域名在该实例网络上超时，而镜像可访问。

**处理**：切换到可用镜像，并只取满足实验协议的 MUSAN noise 子集；数据卡仍记录真实来源、许可证与文件级溯源。RIR 独立使用 OpenSLR RIRS_NOISES 的 real RIR。

**防复发**：下载阶段加入断点续传、镜像列表、速度显示、安全解压和 asset manifest；将“等价子集是否满足实验目的”写进协议，而不是临时偷换数据。

## 2. PEFT 的 `SEQ_2_SEQ_LM` 给 Whisper 注入了错误参数

**现象**：GPU smoke 在 Whisper 前向时报参数不匹配，调用链里出现 PEFT task wrapper 对 `input_ids` 的处理。

**根因**：Whisper 虽然是 encoder-decoder，但输入是 `input_features`，并不等同于标准文本 Seq2Seq。给 LoRA 配置强行设置 `task_type=SEQ_2_SEQ_LM` 后，PEFT 选择了面向文本的包装类，向 Whisper 传递它不接受的参数。

**修复**：将 `task_type` 设为 `None`，使用通用 `PeftModel`，保留 Whisper 自己的 forward 语义；`gpu_smoke.py` 与正式 `train_lora.py` 同步修改，避免 smoke 与训练配置漂移。

**验证**：真实 `openai/whisper-small` 在 CUDA 上完成一次带 LoRA 的前向、反向和有限梯度检查，`gpu_smoke.json` 为 `passed`。

## 3. 冻结主干 + reentrant checkpoint 导致 loss 没有梯度图

**现象**：修复 PEFT wrapper 后，反向阶段出现 loss 不需要梯度或 checkpoint 输入无 `requires_grad` 的问题。

**根因**：LoRA 冻结绝大多数主干参数，而传统 reentrant gradient checkpointing 要求至少一个输入参与梯度。Whisper 的特征输入本身不求导，于是 checkpoint 认为这段图不需要保存，LoRA 的可训练分支也被切断。

**修复**：显式使用 `gradient_checkpointing_kwargs={"use_reentrant": False}`。非 reentrant 实现能正确追踪冻结主干内部的可训练 LoRA 参数。

**验证**：smoke loss 为有限值 7.5412，221,184 个参数可训练，反向通过；随后五专家与 joint 训练完成。

## 4. SSH 端点多次变化，长任务容易因会话断开丢失

**现象**：AutoDL GPU 资源切换后 SSH 端口变化；本地连接中断，但训练不应被杀死。

**处理**：正式任务运行在命名 `screen` 会话中；流水线把 12 个 stage 的完成 marker 写入 `output/v2/stages/`，训练器再在 stage 内恢复最近 checkpoint。

**防复发**：把恢复分成两层：stage 完成后不重复执行；stage 未完成时从训练 checkpoint 恢复。配置写入 request fingerprint；配置发生实质变化时拒绝把旧 checkpoint 冒充同一次实验。

## 5. “显存占用少”被误解为没有必要使用高端 GPU

**现象**：LoRA smoke 峰值只有 1.056GB，正式训练显存也没有占满 32GB。

**原因**：冻结 Whisper-small 主干、LoRA 低秩参数、较小 batch 和短音频共同降低显存；显存只是容量，不能直接代表 Tensor Core 利用率、训练吞吐或总时长。

**结论**：这套默认配置确实不需要 32GB 才能“装下”，但更快 GPU 仍可缩短五专家、joint 和四路解码的墙钟时间。若目标是省钱，应比较“单价 × 总时长”，而不是只看显存百分比。

## 6. 为什么 base CER 看起来比旧干净集高很多

**现象**：旧版只在干净 AISHELL-1 test 上报告 base CER 约 8.57%；v2 overall base CER 为 40.41%。

**原因**：两个数字的测试分布不同。v2 overall 等权包含 clean、噪声、混响、变速、噪声+混响，且 base 在 fast/slow 与 noisy+reverb 桶分别达到 54.39% 与 58.35%。不能把旧干净集和 v2 多退化 overall 做直接纵向优劣比较。

**防复发**：任何指标旁边必须同时写数据集、split、退化方式、样本数和系统定义。

## 7. Soft fusion 没有像直觉那样优于 hard routing

**现象**：soft overall CER 19.35%，差于 hard 的 17.17%；noisy+reverb 中 soft 插入错误显著增加。

**解释**：adapter 权重的线性组合不保证模型函数输出线性插值。两个各自有效的 LoRA 更新在 decoder 中叠加，可能放大错误 token 倾向。概率“不确定”也不意味着按概率混参数就是贝叶斯模型平均。

**工程处理**：保留 hard、soft、joint 三条路径并如实报告；soft 采用 top-2 稀疏化、低置信回退与有界缓存，但不把这些机制宣传为已经保证精度提升。

## 8. 结果发布不能只上传一个大压缩包

**问题**：仅有 tar 包不方便读者验证模型卡、直接下载部署 adapter 或核对指标；把 checkpoint 全部上传又会造成体积膨胀和恢复状态混乱。

**发布策略**：

- GitHub：源码、测试、文档、聚合结果、运行清单；不提交权重和 checkpoint。
- Hugging Face：五个最终专家 adapter、joint 部署 adapter、路由器、关键 JSON、完整可移植包。
- F 盘：保留 1.395GB 全量 `output/v2`，含 checkpoint 和逐条评测，作为个人可恢复备份。

## 9. 面试时的 90 秒故障案例模板

“我先用真实 Whisper-small + PEFT 做 GPU smoke，发现不是显存问题，而是框架语义问题。PEFT 的文本 Seq2Seq wrapper 会给 Whisper 注入 `input_ids`，修正为通用 PeftModel 后，反向又因为冻结主干和 reentrant checkpoint 的组合断了梯度图。我检查 forward signature、可训练参数和 loss 的 `grad_fn`，最终改成 non-reentrant checkpoint。smoke 能完成真实前反向后才放行全量流水线。为避免 SSH 切换导致重跑，我又用 stage marker、checkpoint 和配置指纹做三层保护。最终五专家、joint、四路 5,000 条评测完整跑通。”

这个说法能回到代码和产物验证，不要扩写成没有发生过的分布式训练或生产事故。

## 10. Hugging Face 镜像上传在 multipart 完成阶段失败

**现象**：官方 Hugging Face 域名从 AutoDL 超时；镜像 API 可以认证并上传分片，但它返回的 multipart completion URL 使用了无法解析的 `hf-mirror.org`，导致已上传分片不能完成提交。

**定位**：14.2MB adapter 的分片进度均到 100%，错误只出现在 `complete_multipart`；远端 DNS 能解析 `hf-mirror.com`，不能解析 `.org`。因此不是 token、文件内容或带宽问题，而是镜像签发的完成地址主机错误。

**修复**：上传客户端只对精确前缀 `https://hf-mirror.org/` 重写为活动的 `https://hf-mirror.com/`，不改路径、查询参数或签名。重试后六个 adapter、路由器和 364,673,111-byte tar 全部完成；随后通过模型 API 重新列举 18 个 v2 文件，并比对 tar、路由器和 adapter 的 LFS SHA-256。

**边界**：这是特定镜像的兼容补丁，不应放进训练代码，也不应做宽泛的 URL 替换。网络恢复后应优先回到官方 endpoint。
