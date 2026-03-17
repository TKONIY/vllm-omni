# `chunk_size_utils.py` -- 动态分块大小计算工具

## 文件概述

`chunk_size_utils.py` 提供了异步分块流式传输场景下的动态分块大小（Initial Chunk Size）计算工具。通过根据系统负载动态调整初始分块大小，在低负载时减小首字延迟（TTFA），在高负载时增大分块以摊薄解码开销。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/chunk_size_utils.py`

## 关键代码解析

### max_ic_for_chunk_size -- 计算最大 IC 值

```python
def max_ic_for_chunk_size(chunk_size: int) -> int:
    """Largest power of 2 strictly less than chunk_size."""
    if chunk_size <= 2:
        return 1
    return 1 << ((chunk_size - 1).bit_length() - 1)
```

计算小于 `chunk_size` 的最大 2 的幂。例如：
- `chunk_size=25` -> `max_ic=16`
- `chunk_size=32` -> `max_ic=16`
- `chunk_size=64` -> `max_ic=32`

### compute_dynamic_initial_chunk_size -- 动态计算 IC

```python
def compute_dynamic_initial_chunk_size(
    active_requests: int,
    max_batch_size: int,
    max_ic: int,
) -> int:
```

根据负载因子（`active_requests / max_batch_size`）从 2 的幂序列 `[2, 4, ..., max_ic]` 中选取合适的 IC 值：

- **低负载**（load_factor 接近 0）: 选择小 IC，如 2，使首帧音频更快产出
- **高负载**（load_factor 接近 1）: 选择大 IC（接近 max_ic），减少 Code2Wav 的频繁调用，提高吞吐

```python
steps = [2, 4, 8, ..., max_ic]  # 2 的幂序列
load_factor = min(active_requests / max_batch_size, 1.0)
idx = int(round(load_factor * (len(steps) - 1)))
return steps[idx]
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `max_ic_for_chunk_size` | 函数 | 计算小于 chunk_size 的最大 2 的幂 |
| `compute_dynamic_initial_chunk_size` | 函数 | 根据负载动态选择初始分块大小 |

## 与其他模块的关系

- **qwen3_tts.py**: `talker2code2wav_async_chunk` 调用这两个函数计算动态 IC
- **fish_speech.py**: `slow_ar_to_dac_decoder_async_chunk` 中也使用分块逻辑（但直接从配置读取 IC）
- **stage_configs/*.yaml**: `codec_chunk_frames` 配置项决定了 `chunk_size` 参数

## 总结

`chunk_size_utils.py` 通过两个简洁的函数实现了负载自适应的分块大小策略。核心思想是：系统空闲时用小分块减少延迟，系统繁忙时用大分块提高吞吐。这种动态调整对流式语音合成场景（如 Qwen3-TTS）的用户体验有显著影响。
