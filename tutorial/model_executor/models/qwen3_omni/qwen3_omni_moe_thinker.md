# `qwen3_omni_moe_thinker.py` — Thinker 模型（多模态理解）

## 文件概述

本文件实现了 Qwen3-Omni MoE 的 Thinker 模型，是所有文件中最大最复杂的一个（约 1400 行）。它负责多模态理解（图像、视频、音频）和文本生成，支持 DeepStack 多尺度视觉特征、音频转录、以及复杂的 MRoPE 位置编码。

## 关键代码解析

### 1. 视觉 Transformer 修复

```python
class Qwen3Omni_VisionTransformer(_Qwen3Omni_VisionTransformer):
    """修复 Qwen2_5_VisionAttention.forward() 的 sequence_lengths 参数兼容性"""
    def forward(self, x, grid_thw):
        # 计算 cu_seqlens 和 sequence_lengths
        sequence_lengths = MMEncoderAttention.maybe_compute_sequence_lengths(...)
        # 逐层前向传播，收集 DeepStack 特征
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = hidden_states + blk.attn(blk.norm1(hidden_states), ...)
            if layer_num in deepstack_visual_indexes:
                hidden_states_list.append(hidden_states)
        # 合并多尺度特征
        hidden_states = self.merger(hidden_states)
        for idx, x_ds in enumerate(hidden_states_list):
            processed_hidden_states_list.append(self.merger_list[idx](x_ds))
        return torch.cat(processed_hidden_states_list, dim=1)
```

### 2. 多模态处理器

```python
class Qwen3OmniMoeThinkerMultiModalProcessor(Qwen2_5OmniThinkerMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        # 音频预处理：补齐到 hop_length 的倍数
        mm_data["audio"] = [pad_to_hop_length(audio, hop_length) for audio in audios]
        # 手动计算 feature_attention_mask
        hf_inputs["feature_attention_mask"] = [torch.ones(num_frame) for ...]
```

### 3. MoE LLM 模型（带中间层捕获）

```python
class Qwen3MoeLLMModel(_Qwen3MoeLLMModel):
    def forward(self, ..., capture_layer_indices=None, return_hidden_states=False, deepstack_input_embeds=None):
        for layer_idx, layer in enumerate(self.layers[...]):
            if capture_set and layer_idx in capture_set:
                captured_hidden_states[str(layer_idx)] = hidden_states.clone()
            hidden_states, residual = layer(positions, hidden_states, residual)
            # DeepStack: 注入多尺度视觉特征
            if deepstack_input_embeds and layer_idx in range(0, len(deepstack_input_embeds)):
                hidden_states = hidden_states + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
```

### 4. MRoPE 位置编码

```python
def get_mrope_input_positions(self, input_tokens, mm_features):
    for offset, modality, data in self.iter_mm_features(mm_features):
        if modality == "image":
            grid_indices = np.indices((grid_t, grid_h, grid_w))
            llm_pos_ids_list.append(grid_indices.reshape(3, -1) + st_idx)
        elif modality == "video" and data["use_audio_in_video"]:
            pos_ids, _ = self._compute_interleaved_positions(st_idx, data)
        elif modality == "audio":
            audio_pos = np.broadcast_to(np.arange(audio_tokens), (3, audio_tokens))
```

### 5. 语音转录支持

```python
class Qwen3OmniMoeThinkerForConditionalGeneration(..., SupportsTranscription):
    @classmethod
    def get_generation_prompt(cls, audio, stt_config, ...):
        instruction = "Transcribe" if task_type == "transcribe" else "Translate"
        instruction += " this audio"
        if full_lang_name: instruction += f" into {full_lang_name}"
```

支持 18 种语言的语音转录和翻译。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3Omni_VisionTransformer` | 类 | 视觉编码器（含 DeepStack） |
| `Qwen3OmniMoeThinkerProcessingInfo` | 类 | 多模态处理信息 |
| `Qwen3OmniMoeThinkerMultiModalProcessor` | 类 | 多模态输入处理器 |
| `Qwen3MoeLLMModel` | 类 | MoE LLM（带中间层捕获） |
| `Qwen3MoeLLMForCausalLM` | 类 | MoE 因果语言模型 |
| `Qwen3OmniMoeConditionalGenerationMixin` | Mixin | 多模态输入处理 |
| `Qwen3OmniMoeThinkerForConditionalGeneration` | 类 | Thinker 主模型 |
| `get_mrope_input_positions()` | 方法 | MRoPE 位置计算 |
| `iter_mm_features()` | 方法 | 多模态特征迭代器 |
| `_compute_interleaved_positions()` | 方法 | 音视频交错位置计算 |

## 与其他模块的关系

- **被引用**: `qwen3_omni.py` 和 `qwen3_omni_moe_talker.py` 引用其类和 Mixin
- **继承**: vLLM 上游 `Qwen2_5OmniThinkerMultiModalProcessor`
- **依赖**: `qwen2_5_omni_thinker.py` 的 `Qwen2_5OmniConditionalGenerationMixin`

## 总结

Thinker 是 Qwen3-Omni MoE 最复杂的组件。相比 Qwen2.5-Omni Thinker，主要增强包括：(1) DeepStack 多尺度视觉特征注入；(2) MoE 架构替代 Dense Transformer；(3) 中间层隐藏状态捕获（供 Talker 使用）；(4) 更完善的多模态处理器（音频帧对齐、use_audio_in_video 推断）；(5) 语音转录/翻译支持。
