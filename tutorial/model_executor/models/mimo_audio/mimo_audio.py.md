# `mimo_audio.py` — MiMo-Audio 顶层模型

## 文件概述

MiMo-Audio 的顶层模型入口，约 910 行。包含多模态处理器链（音频编码、交错重排）、两阶段模型路由（fused_thinker_talker / code2wav）、以及 preprocess 钩子。

## 关键代码解析

### 1. 交错重排函数

```python
def interleave_5_and_5_in_span(input_ids, ...):
    """在 codec span 内将文本 token 和填充 token 交错排列"""
    # 找到 [span_start] ... [span_end] 区间
    # 将区间内的文本 token 按 5 个一组排列
    # 每组之间插入 5 个 pad token
    # 保持总长度不变
```

### 2. 音频数据解析器

```python
class MiMoAudioDataParser(MultiModalDataParser):
    def _parse_audio_data(self, data):
        """将原始音频编码为 codec codes"""
        wav_mono = wav_tensor.mean(dim=0)  # 转单声道
        audio_codes = self.mimo_tokenizer.encode(audio=(wav_mono, self.target_sr))
        return AudioProcessorItems(new_audios)
```

### 3. 模型初始化

```python
class MiMoAudioForConditionalGeneration(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        if self.model_stage == "fused_thinker_talker":
            self.has_preprocess = True
            self.set_custom_preprocess(self.fused_thinker_talker_preprocess)
            self.fused_thinker_talker = init_vllm_registered_model(
                architectures=["MiMoAudioLLMModel"])
        elif self.model_stage == "code2wav":
            self.token2wav = init_vllm_registered_model(
                architectures=["MiMoAudioToken2WavModel"])
```

### 4. Preprocess 钩子

```python
def fused_thinker_talker_preprocess(self, input_ids, input_embeds, **info_dict):
    # 1. 交错重排 input_ids
    prompt_ids = interleave_5_and_5_in_span(input_ids.tolist())
    # 2. 处理多模态特征
    mm_embeddings = self.fused_thinker_talker.embed_multimodal(**mm_kwargs_group)
    # 3. 在 empty token 位置注入音频嵌入
    input_embeds = self.fused_thinker_talker.embed_input_ids(
        prompt_ids, multimodal_embeddings=mm_embeddings,
        is_multimodal=(prompt_ids == empty_token_id))
```

### 5. 前向传播

```python
def forward(self, input_ids, positions, ...):
    if self.model_stage == "fused_thinker_talker":
        next_speech_tokens, text_hidden_states = self.generate_codes(...)
        return OmniOutput(text_hidden_states=..., multimodal_outputs={"code_predictor_codes": ...})
    if self.model_stage == "code2wav":
        audio_tensor = self.generate_audio(code)
        return OmniOutput(multimodal_outputs={"model_outputs": audio_tensor})
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `interleave_5_and_5_in_span()` | 函数 | span 内文本/填充交错 |
| `MiMoAudioLLMProcessingInfo` | 类 | 多模态处理信息 |
| `MiMoAudioDataParser` | 类 | 音频数据解析器（含 tokenizer） |
| `MiMoAudioLLMMultiModalProcessor` | 类 | 多模态处理器 |
| `MiMoAudioForConditionalGeneration` | 类 | 顶层模型（阶段路由 + preprocess） |

## 与其他模块的关系

- 依赖 `mimo_audio_llm.py` 的 `MiMoAudioLLMModel`（fused_thinker_talker）
- 依赖 `mimo_audio_code2wav.py` 的 `MiMoAudioToken2WavModel`
- 使用 `CustomProcessMixin` 设置自定义 preprocess 钩子
- 音频编码使用 `MiMoAudioTokenizerWorker`

## 总结

MiMo-Audio 的顶层文件协调了音频预处理（交错重排）、多模态嵌入注入、两阶段推理路由等关键流程。交错机制是其独特设计，使文本和音频 token 在同一序列中高效交互。
