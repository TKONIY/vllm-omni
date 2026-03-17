# `mimo_audio.py` -- MiMo-Audio LLM 到 Code2Wav 处理器

## 文件概述

`mimo_audio.py` 实现了 MiMo-Audio 语音对话模型的 Stage 0（融合 Thinker+Talker）到 Stage 1（Code2Wav）的数据转换。MiMo-Audio 使用 8 层 RVQ 编码（每帧 4 个分量），处理器需要将这些编码按列优先顺序展平，并添加填充向量。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/mimo_audio.py`

## 关键代码解析

### 列优先展平 -- prepend_and_flatten_colmajor

MiMo-Audio 的编码张量形状为 `[batch, 1, 8, 4]`（8 层 RVQ，每层 4 个分量），需要先添加一行填充向量，然后按列优先顺序展平：

```python
def prepend_and_flatten_colmajor(x: torch.Tensor, pad_vec: torch.Tensor) -> torch.Tensor:
    pad_expand = pad_vec.view(*([1] * (x.dim() - 2)), 1, x.size(-1)).expand(...)
    y = torch.cat([pad_expand, x], dim=-2)  # (..., R+1, C)
    # 列优先展平：先转置最后两维，再 reshape
    y_col_major = y.permute(*range(y.dim() - 2), -1, -2).reshape(-1)
    return y_col_major
```

填充向量使用特殊 token ID `TALKER_CODEC_PAD_TOKEN_ID = 8292`。

### llm2code2wav -- 同步处理器

```python
def llm2code2wav(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False):
```

处理流程：
1. 提取 `code_predictor_codes`（形状 `[seq_batch_size, 1, 8, 4]`）
2. 过滤全零帧：`is_all_zero = (codec_codes == 0).all(dim=(1, 2, 3))`
3. 对每帧添加填充向量并列优先展平
4. 封装为 `OmniTokensPrompt`

### llm2code2wav_async_chunk -- 异步分块处理器

```python
def llm2code2wav_async_chunk(transfer_manager, pooling_output, request, is_finished=False):
```

核心逻辑：
1. 从 `pooling_output` 中提取 `code_predictor_codes`
2. 验证张量形状为 `(*, 8, 4)`
3. 添加填充向量并展平为列表
4. 累积到 `transfer_manager.code_prompt_token_ids[request_id]`
5. 按固定 `chunk_size=10` 和 `left_context_size=10` 分块发送

```python
chunk_size = left_context_size = 10
transfer_manager.code_prompt_token_ids[request_id].append(code_list)
length = len(transfer_manager.code_prompt_token_ids[request_id])
chunk_length = length % chunk_size
if chunk_length != 0 and not is_finished:
    return None  # 等待凑满
```

### 特殊 Token

```python
TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294
```

这些常量从 `config_mimo_audio.py` 导入，用于 RVQ 编码序列的边界标记。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `llm2code2wav` | 函数 | 同步处理：LLM 输出 -> Code2Wav 输入 |
| `llm2code2wav_async_chunk` | 函数 | 异步分块处理：逐步流式传输 |
| `prepend_and_flatten_colmajor` | 函数 | 添加填充向量 + 列优先展平 |
| `_make_finished_sentinel` | 函数 | 生成结束标记 payload |

## 与其他模块的关系

- **stage_configs/mimo_audio.yaml**: `custom_process_input_func` 引用同步版
- **stage_configs/mimo_audio_async_chunk.yaml**: 引用异步版
- **models/mimo_audio/config_mimo_audio.py**: 提供 `TALKER_CODEC_PAD_TOKEN_ID`
- **models/mimo_audio/**: LLM 模型输出 `code_predictor_codes`

## 总结

`mimo_audio.py` 的核心特色是列优先展平（Column-Major Flatten）：MiMo-Audio 的 8 层 RVQ 编码需要按列优先顺序展开（先遍历 codebook 再遍历时间步），与 Qwen 系列的行优先展平不同。异步模式使用固定的 chunk_size=10，设计较为简洁。
