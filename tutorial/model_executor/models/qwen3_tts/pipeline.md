# `pipeline.yaml` — 多阶段流水线配置

## 文件概述

本文件定义了 Qwen3-TTS 的两阶段流水线配置，包括 Talker（AR 生成）和 Code2Wav（波形生成）阶段的调度参数、引擎配置、连接器设置和默认采样参数。

## 关键代码解析

### 1. 全局配置

```yaml
model_type: qwen3_tts
async_chunk: true   # 启用异步分块
```

### 2. Talker 阶段（Stage 0）

```yaml
- stage_id: 0
  model_stage: qwen3_tts
  stage_type: llm
  worker_type: ar                          # 自回归工作器
  scheduler_cls: OmniARScheduler           # AR 调度器
  engine_args:
    model_arch: Qwen3TTSTalkerForConditionalGeneration
    enforce_eager: false                   # 允许 CUDA Graph
    gpu_memory_utilization: 0.3
    max_num_batched_tokens: 512
    max_model_len: 4096
  default_sampling_params:
    temperature: 0.9                       # TTS 需要一定随机性
    top_k: 50
    repetition_penalty: 1.05              # 避免重复
    stop_token_ids: [2150]                # codec EOS
```

### 3. Code2Wav 阶段（Stage 1）

```yaml
- stage_id: 1
  model_stage: code2wav
  stage_type: llm
  worker_type: generation                  # 生成工作器（非 AR）
  scheduler_cls: OmniGenerationScheduler
  final_output: true
  final_output_type: audio
  engine_args:
    model_arch: Qwen3TTSCode2Wav
    enforce_eager: true                    # Code2Wav 使用 eager 模式
    gpu_memory_utilization: 0.1            # 解码器占用少
    max_model_len: 32768
  default_sampling_params:
    temperature: 0.0                       # 确定性解码
    max_tokens: 65536
```

### 4. 共享内存连接器

```yaml
connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      shm_threshold_bytes: 65536
      codec_streaming: true                # 启用 codec 流式传输
      codec_chunk_frames: 25              # 每块 25 帧
      codec_left_context_frames: 25       # 25 帧左上下文
```

## 核心配置项

| 配置项 | 说明 |
|--------|------|
| `async_chunk: true` | 异步分块模式 |
| `worker_type: ar` | Talker 使用自回归工作器 |
| `worker_type: generation` | Code2Wav 使用生成工作器 |
| `codec_streaming: true` | 流式 codec 传输 |
| `codec_chunk_frames: 25` | 每块 25 帧（~1秒 @25Hz） |
| `custom_process_next_stage_input_func` | 自定义的阶段间数据处理函数 |

## 与其他模块的关系

- **Stage 0** 使用 `qwen3_tts_talker.py` 中的模型
- **Stage 1** 使用 `qwen3_tts_code2wav.py` 中的模型
- **连接器** 通过共享内存传递 codec 流

## 总结

`pipeline.yaml` 是 Qwen3-TTS 在 vLLM-Omni 框架中的部署蓝图。它定义了两个阶段的资源分配（GPU 内存、batch 大小）、采样策略、以及异步流式传输配置。关键特性是 codec 流式传输（每 25 帧传递一次），实现低延迟的实时语音合成。
