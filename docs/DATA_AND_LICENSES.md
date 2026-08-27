# 数据、磁盘预算与许可证

## 默认正式链路

| 数据 | 用途 | 许可证/条款 | 默认行为 |
|---|---|---|---|
| AISHELL-1 | 中文 ASR 干净语音与文本 | Apache-2.0 | 优先使用 AutoDL 公共盘归档 |
| MUSAN SLR17 | noisy/noisy_reverb 的真实噪声 | CC BY 4.0 | 下载约 11GB；默认只索引 `musan/noise` |
| RIRS_NOISES SLR28 | reverb/noisy_reverb 的真实 RIR | Apache-2.0 | 下载约 1.3GB；按官方 `rir_list` 精确索引，避免把 isotropic noise 当 RIR |

官方页面：

- MUSAN: https://www.openslr.org/17/
- RIRS_NOISES: https://www.openslr.org/28/
- AISHELL-1: https://www.openslr.org/33/

manifest 会写 `source` 与 `source_license`，但代码中的标签不能替代发布前法律核验。发布 GitHub/Hugging Face 时默认只发布代码、manifest schema、指标和模型 adapter，不重新分发原始音频。

## 为什么不默认下载 DNS Challenge

DNS Challenge 5 官方仓库给出的展开规模约 1TB，仅 noise_fullband 就约 58GB、impulse responses 约 5.9GB。把全量 DNS 塞进 200GB 数据盘会挤压训练 checkpoint 和退化数据空间，也会浪费付费下载时间。

因此 v2 的默认“真实数据最小闭环”是 MUSAN + RIRS_NOISES。若远端已有 DNS/WHAM 裁剪集，可通过重复 spec 接入，并保留各自来源：

```bash
python -m src.asset_manifest \
  --noise-spec "MUSAN::CC BY 4.0::/data/musan/noise" \
  --noise-spec "DNS5-noise-subset::upstream terms::/data/dns_noise_subset" \
  --noise-spec "WHAM::upstream terms::/data/wham_noise" \
  --rir-spec "OpenSLR RIRS_NOISES real RIR::Apache-2.0::/data/RIRS_NOISES/real_rirs_isotropic_noises/rir_list" \
  --output-dir /data/assets
```

DNS Challenge 官方仓库：https://github.com/microsoft/DNS-Challenge

不同 DNS 子数据集保留原始条款，不能笼统标成同一个许可证。`source_license` 应按实际下载子集填写。

## 通用干净语音扩展

`data_augmentation` 接受统一 JSONL，不绑定 AISHELL。额外中文/多语数据只要提供这些字段即可：

```json
{"audio_path":"/abs/example.wav","text":"转写文本","speaker":"spk1","utterance_id":"utt1","source_id":"stable-unique-id","dataset":"dataset-name","source_license":"license","split":"train"}
```

训练、验证、测试 source ID 必须互斥。新增数据不能只混进训练而不建立相应的独立泛化 test；否则只能证明“见过更多数据”，不能证明跨域能力。

## 磁盘清理边界

可清理：

- 已确认不再续训的 `checkpoint-*`；
- Hugging Face 未引用的旧 snapshot；
- `evaluation.partial.jsonl` 在最终 CSV、summary、manifest 校验完成并备份后；
- 原始下载压缩包在解压校验、结果备份完成后。

不可在训练中清理：当前 checkpoint、optimizer state、请求指纹、stage marker、真实退化 manifest。正式清盘应在模型/源码/报告已经拉回本地并校验 SHA-256 之后执行。
