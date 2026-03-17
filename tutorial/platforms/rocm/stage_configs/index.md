# rocm/stage_configs/ -- ROCm 阶段配置

## 目录概述

该目录存放 ROCm 平台专用的阶段（stage）配置 YAML 文件，用于定义多阶段 Omni 模型推理流水线的各阶段参数。

## 配置文件列表

| 文件 | 模型 | 验证环境 | 说明 |
|------|------|---------|------|
| `qwen2_5_omni.yaml` | Qwen2.5-Omni | 2x H100-80G GPU | 3 阶段流水线（thinker + talker + code2wav） |
| `qwen3_omni_moe.yaml` | Qwen3-Omni-MoE | 2x H100-80G GPU | 3 阶段 MoE 流水线 |

## 通用配置结构

每个 YAML 文件包含以下核心结构：

### stage_args：阶段定义

```yaml
stage_args:
  - stage_id: 0          # 阶段 ID
    runtime:
      devices: "0"       # 使用的设备编号
      max_batch_size: 1   # 最大批处理大小
    engine_args:
      model_stage: thinker      # 模型阶段类型
      model_arch: ...            # 模型架构名
      worker_type: ar            # Worker 类型: ar 或 generation
      scheduler_cls: ...         # 调度器类
      gpu_memory_utilization: 0.8 # GPU 内存使用率
      engine_output_type: latent  # 输出类型: latent/text/audio
```

### runtime：流水线运行时配置

```yaml
runtime:
  enabled: true
  defaults:
    window_size: -1       # -1 表示等待上游完全完成再触发下游
    max_inflight: 1       # 每个阶段串行处理
  edges:
    - from: 0             # 定义阶段间数据流
      to: 1
      window_size: -1
```

### ROCm 与 CUDA 配置差异

- ROCm 配置使用 `enforce_eager: true`（目前仅支持 eager 模式）
- 配置了 `max_num_batched_tokens: 32768`
- 设备编号与 ROCm 可见设备对应

## 三阶段流水线说明

所有配置都采用统一的三阶段架构：

1. **Stage 0 -- Thinker**：多模态理解 + 文本生成，输出 hidden states（latent）
2. **Stage 1 -- Talker**：接收 hidden states，生成 codec codes
3. **Stage 2 -- Code2Wav**：将 codec codes 转换为音频波形

阶段间通过 `engine_input_source` 指定上游，通过 `custom_process_input_func` 定义数据转换函数。

## 总结

ROCm 阶段配置与默认 CUDA 配置在结构上完全一致，主要差异在于设备分配和性能参数的调整。所有配置均已在 H100 GPU 上验证通过。
