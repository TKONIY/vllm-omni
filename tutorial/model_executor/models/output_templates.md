# `output_templates.py` -- 统一模型输出数据结构 OmniOutput

## 文件概述

`output_templates.py` 定义了 `OmniOutput`，这是 vllm-omni 所有模型阶段的统一输出数据结构。通过 `NamedTuple` 实现，它将文本隐藏状态、多模态输出、中间张量和下一个 token ID 统一封装。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/models/output_templates.py`

## 关键代码解析

```python
from typing import NamedTuple
import torch
from vllm.sequence import IntermediateTensors

class OmniOutput(NamedTuple):
    """Output from the merged Omni model containing both text and audio."""

    text_hidden_states: torch.Tensor
    multimodal_outputs: dict | None = None
    intermediate_tensors: IntermediateTensors | None = None
    next_token_id: torch.Tensor | None = None
```

### 字段解析

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text_hidden_states` | `torch.Tensor` | (必填) | 文本隐藏状态，用于后续的 token 预测或传递给下游阶段 |
| `multimodal_outputs` | `dict \| None` | `None` | 多模态输出字典，键值对因模型而异（如 `"latent"`, `"code_predictor_codes"`, `"audio_codes"` 等） |
| `intermediate_tensors` | `IntermediateTensors \| None` | `None` | 中间张量，用于流水线并行（Pipeline Parallel）场景下的跨设备传输 |
| `next_token_id` | `torch.Tensor \| None` | `None` | 预测的下一个 token ID，某些模型（如 MTP 模型）在 forward 中直接产生 |

### 设计选择

使用 `NamedTuple` 而非 `dataclass` 的原因：
- **不可变性**: `NamedTuple` 实例创建后不可修改，避免意外篡改输出
- **元组兼容**: 可以像普通元组一样解包 `text_hs, mm_out, inter, next_id = output`
- **轻量**: 无额外的 `__init__`、`__repr__` 等方法开销

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OmniOutput` | NamedTuple | 统一的模型输出数据结构 |

## 与其他模块的关系

- **models/ 中的所有模型**: 各模型的 `forward` 方法返回 `OmniOutput` 实例
- **worker/**: Worker 层接收 `OmniOutput` 并提取所需字段
- **stage_input_processors/**: 下游阶段的处理器从 `multimodal_outputs` 中提取数据构造下一阶段输入

### multimodal_outputs 字典的常见键

| 键名 | 使用模型 | 说明 |
|------|----------|------|
| `"latent"` | Qwen2.5-Omni, Qwen3-Omni, MammothModa2 | 隐藏状态张量 |
| `"code_predictor_codes"` | Qwen3-Omni Talker, MiMo-Audio | RVQ 编码预测 |
| `"audio_codes"` | Qwen3-TTS, Fish Speech | 音频编码序列 |
| `"0"`, `"24"` | Qwen3-Omni Thinker | 不同层的隐藏状态 |
| `"tts_bos_embed"` 等 | Qwen3-Omni Thinker | TTS 特殊 token 的嵌入向量 |

## 总结

`OmniOutput` 是一个简洁但关键的数据结构，它用四个字段统一了所有模型阶段的输出格式。`multimodal_outputs` 字典的灵活设计使得不同模型可以传递任意类型的多模态数据，而 `text_hidden_states` 保证了文本推理管线的一致性。
