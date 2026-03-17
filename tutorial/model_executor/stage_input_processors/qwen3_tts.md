# `qwen3_tts.py` -- Qwen3-TTS Talker 到 Code2Wav 处理器

## 文件概述

`qwen3_tts.py` 实现了 Qwen3-TTS 模型的 Talker 阶段到 Code2Wav 阶段的数据转换。该处理器的异步分块版本是整个项目中最精细的分块策略实现，包含动态 IC（Initial Chunk）、ref_code 前缀注入、以及负载自适应等特性。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/qwen3_tts.py`

## 关键代码解析

### talker2code2wav -- 同步处理器

```python
def talker2code2wav(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False):
```

处理流程：
1. 提取 `audio_codes`（形状 `[num_frames, Q]`，Q=16 quantizers）
2. 过滤全零帧（EOS/无效步）
3. 如果存在 `ref_code`（参考语音编码），拼接到 `audio_codes` 前面
4. Codebook-major 展平：`transpose(0,1).reshape(-1)`

```python
audio_codes = output.multimodal_output["audio_codes"].to(torch.long)
valid_mask = audio_codes.any(dim=1)
audio_codes = audio_codes[valid_mask]

ref_code = output.multimodal_output.get("ref_code")
if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
    audio_codes = torch.cat([ref_code.to(audio_codes.device), audio_codes], dim=0)
    ref_code_len = int(ref_code.shape[0])
```

### talker2code2wav_async_chunk -- 异步分块处理器

```python
def talker2code2wav_async_chunk(transfer_manager, pooling_output, request, is_finished=False):
```

这是整个项目中最复杂的异步处理器，核心特性如下：

**1. 动态 Initial Chunk (IC) 计算**

```python
from .chunk_size_utils import compute_dynamic_initial_chunk_size, max_ic_for_chunk_size

if not per_request_override:
    max_ic = max_ic_for_chunk_size(chunk_size)
    active = sum(1 for v in transfer_manager.code_prompt_token_ids.values() if len(v) > 0)
    capacity = getattr(transfer_manager, "scheduler_max_num_seqs", 1)
    initial_chunk_size = compute_dynamic_initial_chunk_size(active, capacity, max_ic)
```

- 低负载时选小 IC（更快的首帧音频输出）
- 高负载时选大 IC（减少 Code2Wav 调用次数）
- 支持 per-request override 通过 `additional_information.entries["initial_codec_chunk_frames"]`

**2. 两阶段分块策略**

```python
in_initial_phase = initial_chunk_size > 0 and initial_chunk_size < chunk_size and length < chunk_size

if in_initial_phase:
    # IC 阶段：每 initial_chunk_size 帧发送一次，左上下文逐渐增大
    if not finished and length % initial_chunk_size != 0:
        return None
    context_length = ...
else:
    # 正常阶段：按 chunk_size 发送，偏移量对齐 IC 阶段
    initial_coverage = ((chunk_size - 1) // initial_chunk_size) * initial_chunk_size
    adjusted = length - initial_coverage
    if not finished and adjusted % chunk_size != 0:
        return None
```

**3. ref_code 首窗口注入**

```python
if transfer_manager.put_req_chunk[request_id] == 0:
    ref_code = request_payload.pop(request_id, None)
    if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
        ref_frames = ref_code.tolist()
        window_frames = ref_frames + window_frames
        left_context_size += len(ref_frames)
```

仅在第一个解码窗口前插入参考语音编码，后续窗口不再重复。

**4. 帧提取与累积**

```python
def _extract_last_frame(pooling_output):
    audio_codes = pooling_output.get("audio_codes")
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        return frame.to(torch.long).reshape(-1)
```

每步推理从 `pooling_output` 中提取最后一帧编码，追加到累积缓冲区。

### 输出格式

```python
return {
    "code_predictor_codes": codebook_major_flat_codes,  # [Q * window_frames]
    "left_context_size": int,     # 左上下文帧数
    "finished": torch.tensor,     # 是否结束
}
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `talker2code2wav` | 函数 | 同步：Talker -> Code2Wav |
| `talker2code2wav_async_chunk` | 函数 | 异步分块：动态 IC + ref_code 注入 |
| `_extract_last_frame` | 函数 | 从 pooling 输出中提取最后一帧编码 |

## 分块策略可视化

假设 `chunk_size=25`, `initial_chunk_size=4`:

```
帧序号:  [1-4] [5-8] [9-12] [13-16] [17-20] [21-24] [25-49]  [50-74] ...
发送:     IC    IC    IC      IC      IC      IC     chunk_25  chunk_25
上下文:   0帧   4帧   8帧    12帧    16帧    20帧    25帧      25帧

         <-------- IC 阶段 ----------->  <--- 正常阶段 --->
```

## 与其他模块的关系

- **stage_configs/qwen3_tts.yaml**: `custom_process_next_stage_input_func` 引用异步版
- **stage_configs/qwen3_tts_no_async_chunk.yaml**: `custom_process_input_func` 引用同步版
- **chunk_size_utils.py**: 提供 `max_ic_for_chunk_size` 和 `compute_dynamic_initial_chunk_size`
- **models/qwen3_tts/**: Talker 输出 `audio_codes` 和 `ref_code`

## 总结

`qwen3_tts.py` 是 vllm-omni 异步流式处理的最佳实践，其异步分块处理器融合了三项关键优化：动态 IC 根据系统负载自适应调整首帧延迟与吞吐的平衡、ref_code 首窗口注入实现零样本语音克隆、以及 IC/Normal 两阶段平滑切换确保编码连续性。同步版本则提供了简洁的全量传输方案。
