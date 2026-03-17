# `qwen2_5_omni.py` -- Qwen2.5-Omni Thinker 到 Talker 处理器

## 文件概述

`qwen2_5_omni.py` 实现了 Qwen2.5-Omni 模型的 Thinker 阶段到 Talker 阶段的数据转换。它从 Thinker 的输出中提取隐藏状态，分割为 prompt 嵌入和生成结果嵌入，然后构建 Talker 阶段所需的输入（包括特殊的 codec token 序列）。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/qwen2_5_omni.py`

## 关键代码解析

### 特殊 Token 定义

```python
TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294
```

Talker 阶段的输入 token 序列格式为：
```
[START] [PAD * prompt_len] [END]
```

### thinker2talker 函数

```python
def thinker2talker(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False):
```

处理流程：

1. **提取 Thinker 输出**

```python
latent = output.multimodal_output["latent"]
thinker_hidden_states = latent.clone().detach()
```

2. **分割隐藏状态**

```python
additional_information = {
    "thinker_result": thinker_hidden_states[prompt_token_ids_len:].to(torch.float32),
    "prompt_embeds": thinker_hidden_states[:prompt_token_ids_len].to(torch.float32),
    "prompt_token_ids": prompt_token_ids,
    "thinker_output_token_ids": thinker_output_ids,
}
```

- `prompt_embeds`: prompt 部分的隐藏状态
- `thinker_result`: 生成部分的隐藏状态

3. **构建 Talker 输入**

```python
OmniTokensPrompt(
    prompt_token_ids=[TALKER_CODEC_START_TOKEN_ID]
        + [TALKER_CODEC_PAD_TOKEN_ID] * len(prompt_token_ids)
        + [TALKER_CODEC_END_TOKEN_ID],
    additional_information=additional_information,
    multi_modal_data=multi_modal_data[...] if requires_multimodal_data else None,
)
```

Talker 的 token 序列长度 = prompt 长度 + 2（START + END），其中 PAD token 作为占位符，实际嵌入由 `additional_information` 中的 `prompt_embeds` 和 `thinker_result` 提供。

### 多模态数据传递

当 `requires_multimodal_data=True` 时，Thinker 阶段接收的多模态数据（如音频特征）会被传递给 Talker，通过 `request_id` 进行匹配：

```python
multi_modal_data = {
    thinker_output.request_id: p.get("multi_modal_data", None)
    for thinker_output, p in zip(thinker_outputs, prompt)
}
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `thinker2talker` | 函数 | Thinker 输出 -> Talker 输入转换（同步） |
| `TALKER_CODEC_PAD_TOKEN_ID` | 常量 | 8292 |
| `TALKER_CODEC_START_TOKEN_ID` | 常量 | 8293 |
| `TALKER_CODEC_END_TOKEN_ID` | 常量 | 8294 |

## 与其他模块的关系

- **stage_configs/qwen2_5_omni.yaml**: `custom_process_input_func` 引用 `qwen2_5_omni.thinker2talker`
- **models/qwen2_5_omni/**: Thinker 输出 `latent` 隐藏状态；Talker 消费 `additional_information`
- **qwen3_omni.py**: Qwen3-Omni 的 `thinker2talker` 设计类似但更复杂（支持异步分块和多层隐藏状态）

## 总结

`qwen2_5_omni.py` 实现了 Qwen2.5-Omni 的 Thinker->Talker 阶段转换，核心是将 Thinker 的隐藏状态分割为 prompt 嵌入和生成结果嵌入，并构建以 codec 特殊 token 为骨架的 Talker 输入序列。相比 Qwen3-Omni 版本，它更简洁，只支持同步模式且传递单一的 `latent` 隐藏状态。
