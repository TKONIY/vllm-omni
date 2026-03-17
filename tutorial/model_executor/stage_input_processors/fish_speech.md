# `fish_speech.py` -- Fish Speech S2 Pro 编解码处理器

## 文件概述

`fish_speech.py` 实现了 Fish Speech S2 Pro 模型的 Slow AR 阶段到 DAC Decoder 阶段的数据转换。提供同步和异步分块两种模式：同步模式等待 Slow AR 完成后批量传输，异步模式逐帧流式传输编码数据。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/fish_speech.py`

## 关键代码解析

### 编码格式

Fish Speech 使用 10 个 codebook（1 语义 + 9 残差）的多层 RVQ 编码：

```python
_NUM_CODEBOOKS = 10  # 1 semantic + 9 residual
```

### slow_ar_to_dac_decoder -- 同步处理器

```python
def slow_ar_to_dac_decoder(stage_list, engine_input_source, ...):
    """等待 Slow AR 完成后，将所有编码传递给 DAC 解码器"""
    audio_codes = out.multimodal_output["audio_codes"].to(torch.long)
    valid_mask = audio_codes.any(dim=1)        # 过滤零填充帧
    audio_codes = audio_codes[valid_mask]
    # Codebook-major 展平: [num_codebooks * num_frames]
    codec_codes = audio_codes.transpose(0, 1).cpu().reshape(-1).tolist()
```

### slow_ar_to_dac_decoder_async_chunk -- 异步分块处理器

```python
def slow_ar_to_dac_decoder_async_chunk(
    transfer_manager, pooling_output, request, is_finished=False
):
```

核心流程：

1. **逐帧提取**: `_extract_last_frame(pooling_output)` 从每步输出中提取最后一帧编码
2. **帧累积**: 将编码帧追加到 `transfer_manager.code_prompt_token_ids[request_id]`
3. **分块判断**: 根据 `codec_chunk_frames` 和 `initial_codec_chunk_frames` 配置判断是否应该发送
4. **左上下文**: 发送时附带 `codec_left_context_frames` 帧的上下文，确保音频过渡平滑

### 分块策略（两阶段）

```
Initial Phase (初始阶段): length <= chunk_size
  - 按 initial_chunk_size 为单位发送
  - 左上下文为 length - context_length

Normal Phase (正常阶段): length > chunk_size
  - 按 chunk_size 为单位发送
  - 左上下文为 min(length, left_context_size_config + context_length) - context_length
```

### _extract_last_frame 辅助函数

```python
def _extract_last_frame(pooling_output):
    audio_codes = pooling_output.get("audio_codes")
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]  # 取最后一帧
        return frame.to(torch.long).reshape(-1)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `slow_ar_to_dac_decoder` | 函数 | 同步处理：Slow AR -> DAC Decoder |
| `slow_ar_to_dac_decoder_async_chunk` | 函数 | 异步分块处理：逐帧流式传输 |
| `_extract_last_frame` | 函数 | 从 pooling 输出中提取最后一帧编码 |
| `_NUM_CODEBOOKS` | 常量 | 10 (1 语义 + 9 残差) |

## 与其他模块的关系

- **stage_configs/fish_speech_s2_pro.yaml**: `custom_process_next_stage_input_func` 引用异步处理器
- **models/fish_speech/**: 模型输出 `audio_codes` 被此处理器消费
- **qwen3_omni.py**: 同步处理器中复用了 `_validate_stage_inputs` 辅助函数

## 总结

`fish_speech.py` 为 Fish Speech S2 Pro 模型提供了完整的 Slow AR -> DAC Decoder 数据转换，支持同步批处理和异步流式两种模式。异步模式的两阶段分块策略（初始小块 + 正常大块）与 Qwen3-TTS 的策略类似，体现了在首字延迟和吞吐之间的平衡设计。
