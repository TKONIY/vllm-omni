# `qwen3_omni.py` — 统一入口模型

## 文件概述

本文件实现了 `Qwen3OmniMoeForConditionalGeneration`，即 Qwen3-Omni MoE 模型的统一入口。与 Qwen2.5-Omni 类似，它根据 `model_stage` 配置分发到 thinker/talker/code2wav 三个阶段，但增加了 Code Predictor 的 MTP 集成、异步流式解码、以及更精细的阶段间数据传递。

## 关键代码解析

### 1. 特殊 Token 常量

```python
AUDIO_START_TOKEN_ID = 151669   # <|audio_start|>
TALKER_CODEC_PAD_TOKEN_ID = 4196
TALKER_CODEC_BOS_TOKEN_ID = 4197
TALKER_CODEC_EOS_TOKEN_ID = 4198
TALKER_CODEC_NOTHINK_ID = 4203
TALKER_CODEC_THINK_BOS_ID = 4204
```

Qwen3 引入了 "思考模式" token (`THINK_BOS`/`THINK_EOS`/`NOTHINK`)，用于控制 Talker 的生成行为。

### 2. 阶段初始化（使用 `with_hf_config`）

```python
if self.model_stage == "thinker":
    thinker_vllm_config = vllm_config.with_hf_config(
        thinker_config, architectures=["Qwen3OmniMoeThinkerForConditionalGeneration"]
    )
    self.thinker = init_vllm_registered_model(vllm_config=thinker_vllm_config, ...)
```

使用 `with_hf_config()` 创建子配置，确保每个阶段使用自己的 `hf_config`。

### 3. GPU 端 Token 抑制

```python
def _get_talker_suppressed_tokens(self):
    vocab_size = self.config.talker_config.text_config.vocab_size
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    start = vocab_size - 1024
    eos_id = self.config.talker_config.codec_eos_token_id
    for i in range(start, vocab_size):
        if i != eos_id:
            mask[i] = True
    return mask
```

预计算布尔掩码，在 GPU 上通过 `masked_fill_` 高效抑制非法 token。

### 4. Talker 预处理与 MTP

```python
def talker_preprocess(self, input_ids, input_embeds, **info_dict):
    if span_len > 1:  # prefill
        input_ids, input_embeds, update_dict = self.talker_preprocess_prefill(...)
        update_dict["code_predictor_codes"] = torch.zeros(...)
    else:  # decode
        last_talker_hidden, text_step, update_dict = self.talker_preprocess_decode(...)
        update_dict["mtp_inputs"] = last_talker_hidden, text_step
```

Talker 的预处理分为 prefill 和 decode 两个路径，decode 时准备 MTP 输入。

### 5. Code2Wav 异步流式解码

```python
if self.vllm_config.model_config.async_chunk:
    audio_tensors = self.code2wav.chunked_decode_streaming(
        talker_codes, left_context_size=left_context_size, ...)
```

支持异步分块解码，每个请求可以有不同的 `left_context_size`。

### 6. `make_omni_output` 后处理

```python
def make_omni_output(self, model_outputs, **kwargs):
    if self.model_stage == "thinker":
        # 计算 TTS 特殊 token 嵌入（BOS/EOS/PAD）
        thinker_tts_embeds = self.thinker.embed_input_ids(self.tts_tokens)
    elif self.model_stage == "talker":
        # 聚合 code_predictor_codes
        code_predictor_codes = [info.get("code_predictor_codes") for info in info_dicts]
        multimodal_outputs = {"code_predictor_codes": torch.cat(code_predictor_codes, dim=0)}
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3OmniMoeForConditionalGeneration` | 类 | 统一入口模型 |
| `forward()` | 方法 | 多阶段前向传播 |
| `make_omni_output()` | 方法 | 构造 `OmniOutput` |
| `talker_preprocess()` | 方法 | Talker 自定义预处理 |
| `talker_postprocess()` | 方法 | Talker 自定义后处理 |
| `talker_mtp()` | 方法 | 运行 Code Predictor MTP |
| `generate_audio()` | 方法 | Code2Wav 音频生成 |
| `_init_special_tokens_embeddings()` | 方法 | 初始化特殊 token 嵌入 |
| `_thinker_to_talker_prefill()` | 方法 | Thinker→Talker prefill 投影 |
| `compute_logits()` | 方法 | 计算 logits + token 抑制 |

## 与其他模块的关系

- **引用**: `qwen3_omni_moe_thinker.py` 的处理器和 Mixin
- **引用**: `qwen3_omni_moe_talker.py` 的 Talker 模型
- **引用**: `qwen3_omni_code2wav.py` 的 Code2Wav 模型
- **使用**: `CustomProcessMixin` 注册预处理/后处理回调
- **使用**: `OmniOutput` 统一输出容器

## 总结

`qwen3_omni.py` 相比 Qwen2.5-Omni 的统一入口有显著增强：(1) 引入 Code Predictor MTP 用于多层 RVQ 预测；(2) GPU 端布尔掩码高效抑制 token；(3) 支持异步流式解码；(4) 更精细的 thinker→talker 投影（区分文本/多模态，使用 text_projection/hidden_projection）；(5) 阶段间通过 `model_intermediate_buffer` 传递 code predictor 输出。
