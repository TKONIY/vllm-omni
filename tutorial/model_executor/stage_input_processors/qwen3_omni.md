# `qwen3_omni.py` -- Qwen3-Omni MoE Thinker/Talker/Code2Wav 处理器

## 文件概述

`qwen3_omni.py` 是 Qwen3-Omni MoE 模型的核心阶段间处理器，实现了完整的三阶段流水线转换：Thinker -> Talker -> Code2Wav。同时支持同步和异步分块两种模式，是所有处理器中功能最完整、最复杂的一个。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`

## 关键代码解析

### 公共辅助函数

```python
def _ensure_list(x):
    """将 ConstantList / tensor-like 转为 Python list"""
    if hasattr(x, "_x"):
        return list(x._x)    # 处理 vLLM 的 ConstantList

def _validate_stage_inputs(stage_list, engine_input_source):
    """验证阶段输入的合法性并返回上游输出"""
```

`_validate_stage_inputs` 被本文件和其他文件（如 `fish_speech.py`、`qwen3_tts.py`）复用。

### Thinker -> Talker（同步版）

```python
def thinker2talker(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False):
```

与 Qwen2.5-Omni 版本的关键差异：

1. **多层隐藏状态**: 传递第 0 层和第 24 层的隐藏状态
```python
info = {
    "thinker_prefill_embeddings": output.multimodal_output["0"],    # 第 0 层
    "thinker_hidden_states": output.multimodal_output["24"],        # 第 24 层
    "thinker_sequences": prompt_token_ids + output.token_ids,       # 完整序列
    "thinker_input_ids": prompt_token_ids,
    "tts_bos_embed": output.multimodal_output["tts_bos_embed"],     # TTS 特殊嵌入
    "tts_eos_embed": output.multimodal_output["tts_eos_embed"],
    "tts_pad_embed": output.multimodal_output["tts_pad_embed"],
}
```

2. **动态计算 Talker prompt 长度**: 通过 `_compute_talker_prompt_ids_length` 根据对话角色标记计算

### _compute_talker_prompt_ids_length

```python
def _compute_talker_prompt_ids_length(info, device="cuda"):
```

通过分析 token 序列中的 `<|im_start|>` 标记和角色 token（system/user/assistant）来计算 Talker 的 prompt 长度：
- 跳过 system 角色的长度
- 累加所有 user 角色的长度
- assistant 角色固定长度 9（3+4+1+1）

### Thinker -> Talker（异步分块版）

```python
def thinker2talker_async_chunk(transfer_manager, pooling_output, request, is_finished=False):
```

核心逻辑按 `chunk_id` 区分：

**chunk_id == 0（首次调用）**: 传递完整的 prefill 隐藏状态
```python
talker_additional_info = {
    "thinker_prefill_embeddings": pooling_output.get("0").detach().cpu(),
    "thinker_hidden_states": pooling_output.get("24").detach().cpu(),
    "thinker_sequences": all_token_ids,
    "thinker_input_ids": prompt_token_ids,
    "tts_bos_embed": ..., "tts_eos_embed": ..., "tts_pad_embed": ...,
    "finished": torch.tensor(is_finished, dtype=torch.bool),
}
```

特殊处理：如果 `request_payload` 中已有缓存（说明是分块 prefill 的第二次调用），将两次的嵌入拼接。

**chunk_id > 0（后续调用）**: 只传递增量的 decode 嵌入
```python
talker_additional_info = {
    "finished": torch.tensor(is_finished, dtype=torch.bool),
    "override_keys": ["thinker_decode_embeddings", "thinker_output_token_ids"],
    "thinker_decode_embeddings": pooling_output.get("0").detach().cpu(),
    "thinker_output_token_ids": output_token_ids,
}
```

`override_keys` 指示下游只更新指定字段而非替换整个 payload。

### Talker -> Code2Wav（同步版）

```python
def talker2code2wav(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False):
```

提取 Talker 输出的 RVQ 编码并展平：
```python
seq_len = len(output.token_ids) - 1
codec_codes = (
    output.multimodal_output["code_predictor_codes"][-seq_len:]
    .to(torch.long)
    .transpose(0, 1)    # [seq_len, Q] -> [Q, seq_len]
    .reshape(-1)         # 展平为 [Q * seq_len]
    .tolist()
)
```

### Talker -> Code2Wav（异步分块版）

```python
def talker2code2wav_async_chunk(transfer_manager, pooling_output, request, is_finished=False):
```

流程类似于 Qwen3-TTS 的异步处理器：
1. 从 pooling 输出中提取 `code_predictor_codes`
2. 转为列表并追加到累积缓冲区
3. 按 `codec_chunk_frames` 分块，附带 `codec_left_context_frames` 上下文
4. 返回 `{code_predictor_codes, left_context_size, finished}` 或 `None`（等待凑满）

展平方式为 codebook-major（先转置再 reshape），与 Talker 输出的 `[seq_len, Q]` 格式对应。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `thinker2talker` | 函数 | 同步：Thinker -> Talker |
| `thinker2talker_async_chunk` | 函数 | 异步：Thinker -> Talker（逐步传输） |
| `talker2code2wav` | 函数 | 同步：Talker -> Code2Wav |
| `talker2code2wav_async_chunk` | 函数 | 异步：Talker -> Code2Wav（分块传输） |
| `_compute_talker_prompt_ids_length` | 函数 | 根据角色标记计算 Talker prompt 长度 |
| `_validate_stage_inputs` | 函数 | 验证阶段输入合法性（被多个文件复用） |
| `_ensure_list` | 函数 | ConstantList 转 Python list |

## 与其他模块的关系

- **stage_configs/qwen3_omni_moe.yaml**: 同步版引用 `thinker2talker` 和 `talker2code2wav`
- **stage_configs/qwen3_omni_moe_async_chunk.yaml**: 异步版引用 `thinker2talker_async_chunk` 和 `talker2code2wav_async_chunk`
- **models/qwen3_omni/**: Thinker 输出多层隐藏状态（`"0"`, `"24"`）和 TTS 嵌入；Talker 输出 `code_predictor_codes`
- **engine/OmniEngineCoreRequest**: 异步处理器中使用的请求对象类型
- **fish_speech.py**, **qwen3_tts.py**: 复用了 `_validate_stage_inputs`

## 总结

`qwen3_omni.py` 是 vllm-omni 中最复杂的阶段处理器，完整覆盖了 Qwen3-Omni MoE 三阶段流水线的两个转换点，并为每个转换点提供同步和异步两种模式。异步模式的关键设计包括：首次传输完整 prefill 嵌入 + 后续增量传输 decode 嵌入的分层策略，以及 `override_keys` 增量更新机制。
