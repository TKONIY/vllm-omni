# `cosyvoice3.py` -- CosyVoice3 文本到语音流处理器

## 文件概述

`cosyvoice3.py` 实现了 CosyVoice3 语音合成模型的 Stage 0 (Talker) 到 Stage 1 (Code2Wav/Flow Matching) 的数据转换。该处理器将 Talker 阶段的输出（包括 prompt token IDs 和多模态数据）打包为 Code2Wav 阶段的输入。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/cosyvoice3.py`

## 关键代码解析

### text2flow 函数

```python
def text2flow(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = True,
):
    """Build stage-1 inputs by prefixing stage-0 prompt ids to its outputs."""
```

处理流程：

1. 从 Stage 0 的输出中提取生成的 token IDs 和多模态输出
2. 将 Stage 0 的 prompt token IDs 作为前缀附加到多模态数据中
3. 构造 `OmniTokensPrompt` 作为 Code2Wav 的输入

```python
output_ids = output.token_ids
prefix_ids = source_output.prompt_token_ids
multi_modal_data["prefix_ids"] = prefix_ids
engine_input = OmniTokensPrompt(
    prompt_token_ids=output_ids,
    additional_information=multi_modal_data
)
```

关键设计：`prefix_ids` 被嵌入到 `additional_information` 中传递给 Code2Wav，使其能够利用 prompt 阶段的上下文信息进行更好的语音合成。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `text2flow` | 函数 | Talker 输出 -> Code2Wav 输入转换（同步） |

## 与其他模块的关系

- **stage_configs/cosyvoice3.yaml**: `custom_process_input_func` 引用 `cosyvoice3.text2flow`
- **models/cosyvoice3/**: Code2Wav 模型从 `additional_information` 中读取 `prefix_ids`
- **inputs/data.py**: 使用 `OmniTokensPrompt` 数据结构

## 总结

`cosyvoice3.py` 是一个简洁的同步阶段处理器，核心工作是将 Talker 的 prompt 前缀 ID 和生成的 token IDs 重组为 Code2Wav 所需的输入格式。其设计体现了"前缀传递"的模式，让下游 Flow Matching 模型能够利用上游的完整上下文。
