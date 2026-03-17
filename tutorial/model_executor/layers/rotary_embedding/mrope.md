# `mrope.py` -- 多模态旋转位置编码 OmniMRotaryEmbedding

## 文件概述

`mrope.py` 实现了 `OmniMRotaryEmbedding` 类，这是对 vLLM 原生 `MRotaryEmbedding` 的多模态扩展。该类为不同的多模态场景（图像、视频、音频以及音视频交织输入）提供了专门的位置编码计算方法，是 vllm-omni 支持多模态推理的核心组件。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/layers/rotary_embedding/mrope.py`

## 关键代码解析

### 类继承结构

```python
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

class OmniMRotaryEmbedding(MRotaryEmbedding):
    """Omni-extended MRotaryEmbedding with multimodal position computation."""
```

`OmniMRotaryEmbedding` 继承自 vLLM 的 `MRotaryEmbedding`，保留原有的 Rotary Embedding 计算能力，同时添加了多个类方法来处理不同模型的多模态位置编码。

### 入口方法：get_input_positions_tensor

该方法是位置编码计算的主入口，根据模型类型自动路由到不同的实现：

```python
@classmethod
def get_input_positions_tensor(cls, input_tokens, hf_config, ...):
    if thinker_uses_mrope(hf_config):
        return cls._omni_get_input_positions_tensor(...)    # Qwen2.5-Omni / Qwen3-Omni
    elif hf_config.model_type in ["glm4v", "glm4v_moe"]:
        return cls._glm4v_get_input_positions_tensor(...)   # GLM4V
    else:
        return cls._vl_get_input_positions_tensor(...)      # 通用视觉语言模型
```

### 三维位置编码 (t, h, w)

所有方法都使用三维位置编码（时间/帧 t、高度 h、宽度 w），这是 MRoPE（Multi-dimensional Rotary Position Embedding）的核心思想：

```python
# 图像/视频位置编码示例
t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)
```

### 音视频交织模式 (use_audio_in_video)

Qwen2.5-Omni 特有的模式，将视频帧和音频块交替排列：

```
|V_1 ...    V_n|A_1 ...   A_n|V_n+1 ... V_2n|A_n+1 ... A_2n|...
|vision chunk 1|audio chunk 1|vision chunk 2 |audio chunk 2 |...
```

对应方法 `_omni_get_input_positions_tensor` 中处理了这种交织排列的位置编码计算。

### omni_get_updates_use_audio_in_video

该方法用于在 `use_audio_in_video` 模式下更新 prompt token 序列，将原始的 `<|VIDEO|>` 占位符展开为视频块和音频块交替的形式：

```python
@classmethod
def omni_get_updates_use_audio_in_video(cls, thinker_config, audio_len, video_grid_thw, ...):
    """
    <|video_bos|><|VIDEO|><|video_eos|> =>
    <|video_bos|><|audio_bos|>(... chunks ...)<|audio_eos|><|video_eos|>
    """
```

### 辅助方法

```python
@staticmethod
def _get_llm_pos_ids_for_vision(start_idx, vision_idx, spatial_merge_size, t_index, grid_hs, grid_ws):
    """计算视觉 token 的三维位置 ID，考虑 spatial_merge_size 下采样"""

@staticmethod
def _split_list_into_ranges(lst, interval):
    """将索引列表按时间间隔分组，用于音视频交织分块"""
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OmniMRotaryEmbedding` | 类 | 多模态旋转位置编码扩展类 |
| `get_input_positions` | 类方法 | 入口方法，返回 Python list 格式的位置 |
| `get_input_positions_tensor` | 类方法 | 入口方法，根据模型类型路由到具体实现 |
| `_vl_get_input_positions_tensor` | 类方法 | 通用 VL 模型（如 Qwen2-VL）的位置编码 |
| `_glm4v_get_input_positions_tensor` | 类方法 | GLM4V 模型的位置编码 |
| `_omni_get_input_positions_tensor` | 类方法 | Qwen2.5/3-Omni 的位置编码（含音频支持） |
| `_get_llm_pos_ids_for_vision` | 静态方法 | 视觉 token 三维位置 ID 计算 |
| `_split_list_into_ranges` | 静态方法 | 按时间间隔分组工具 |
| `omni_get_updates_use_audio_in_video` | 类方法 | 音视频交织模式下的 prompt 更新 |

## 与其他模块的关系

- **vllm.model_executor.layers.rotary_embedding.mrope.MRotaryEmbedding**: 父类，提供基础 RoPE 计算
- **vllm.transformers_utils.config.thinker_uses_mrope**: 用于判断模型是否使用 Omni 风格的 MRoPE
- **models/qwen2_5_omni/**, **models/qwen3_omni/**: 这些模型在 prefill 阶段调用该类计算位置编码
- **models/glm_image/**: GLM-Image 模型使用 GLM4V 分支的位置编码

## 总结

`mrope.py` 是 vllm-omni 多模态位置编码的核心实现，通过继承 vLLM 的 `MRotaryEmbedding` 并添加三个特化方法（VL 通用、GLM4V、Omni），实现了对图像、视频、音频及其混合输入的三维位置编码支持。其中最复杂的是 Omni 模式的音视频交织位置编码，这是 Qwen2.5-Omni 模型处理视频中音轨的关键技术。
