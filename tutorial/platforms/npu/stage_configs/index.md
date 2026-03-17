# npu/stage_configs/ -- NPU 阶段配置

## 目录概述

该目录存放 NPU（Ascend）平台专用的阶段配置 YAML 文件。NPU 平台拥有最丰富的配置选项，包含同步模式、异步块传输模式和独立 TTS 配置等多种场景。

## 配置文件列表

| 文件 | 模型 | 验证环境 | 说明 |
|------|------|---------|------|
| `qwen2_5_omni.yaml` | Qwen2.5-Omni | NPU | 3 阶段同步流水线 |
| `qwen3_omni_moe.yaml` | Qwen3-Omni-MoE | 5x A2/A3-64G NPU | 3 阶段 MoE 流水线 |
| `qwen3_omni_moe_async_chunk.yaml` | Qwen3-Omni-MoE | 2x H100-80G GPU | 3 阶段异步块传输流水线 |
| `qwen3_tts.yaml` | Qwen3-TTS | NPU | 2 阶段 TTS 独立流水线 |

## NPU 配置与其他平台的差异

### 1. enforce_eager

NPU 配置中 talker 阶段通常设置 `enforce_eager: true`：
```yaml
enforce_eager: true  # haven't supported talker ACL graph on NPU
```
表明 talker 阶段尚未支持 NPU 的 ACL Graph 加速。

### 2. 张量并行

Qwen3-Omni-MoE 的 thinker 阶段使用多设备并行：
```yaml
devices: "0,1"
tensor_parallel_size: 2
distributed_executor_backend: "mp"
```

### 3. async_chunk 异步块传输模式

`qwen3_omni_moe_async_chunk.yaml` 启用了异步块传输：
```yaml
async_chunk: true
```
该模式下：
- 支持更大的 `max_batch_size`（10）
- 使用 `custom_process_next_stage_input_func` 代替 `custom_process_input_func`
- talker 阶段的 `repetition_penalty` 设为 1.0（不惩罚重复）

### 4. TTS 独立配置

`qwen3_tts.yaml` 是一个 2 阶段的纯 TTS 流水线：
```yaml
stage_args:
  - stage_id: 0   # Talker（文本到 codec codes）
    engine_args:
      model_arch: Qwen3TTSTalkerForConditionalGeneration
      hf_overrides:
        architectures: [Qwen3TTSTalkerForConditionalGeneration]
  - stage_id: 1   # Code2Wav（codec codes 到音频）
    engine_args:
      model_arch: Qwen3TTSCode2Wav
      hf_overrides:
        architectures: [Qwen3TTSCode2Wav]
```

特殊配置：
- 使用 `hf_overrides` 强制指定阶段专用架构
- 配置了 `output_connectors` 和 `input_connectors` 实现共享内存传输
- 包含 `codec_streaming` 相关的流式传输参数
- 定义了 `tts_args.max_instructions_length` 限制指令长度

### 5. 共享内存连接器

TTS 配置使用命名连接器实现阶段间通信：
```yaml
runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 65536
        codec_streaming: true
        codec_chunk_frames: 25
        codec_left_context_frames: 25
```

## 总结

NPU 阶段配置是四个平台中最丰富的，覆盖了同步/异步推理、多设备并行、独立 TTS 流水线等多种场景。这些配置反映了 NPU 平台在 Omni 多模态推理方面的深度适配工作。
