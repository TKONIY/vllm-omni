# xpu/stage_configs/ -- XPU 阶段配置

## 目录概述

该目录存放 Intel XPU 平台专用的阶段配置 YAML 文件，包含针对 Intel Arc GPU 优化的参数设置。

## 配置文件列表

| 文件 | 模型 | 验证环境 | 说明 |
|------|------|---------|------|
| `qwen2_5_omni.yaml` | Qwen2.5-Omni | 2x Intel Arc Pro B60 | 3 阶段同步流水线 |
| `qwen3_omni_moe.yaml` | Qwen3-Omni-MoE | 8x Intel Arc Pro B60 | 3 阶段 MoE 流水线 |

## XPU 配置特点

### 1. Qwen2.5-Omni 配置

```yaml
# 验证环境: 2x Intel Arc Pro B60
stage_args:
  - stage_id: 0   # Thinker
    engine_args:
      gpu_memory_utilization: 0.9  # thinker weight ~16.74GB
      enforce_eager: false
  - stage_id: 1   # Talker
    engine_args:
      gpu_memory_utilization: 0.5  # talker weight ~6.03GB
      enforce_eager: false
  - stage_id: 2   # Code2Wav
    engine_args:
      gpu_memory_utilization: 0.3  # code2wav weight ~1.46GB
      enforce_eager: true
```

XPU 平台的 thinker 和 talker 阶段设置 `enforce_eager: false`（启用图优化），但 code2wav 仍使用 eager 模式。配置中注明了各阶段的权重大小。

### 2. Qwen3-Omni-MoE 配置

```yaml
# 验证环境: 8x Intel Arc Pro B60
stage_args:
  - stage_id: 0   # Thinker
    runtime:
      devices: "0,1,2,3"
    engine_args:
      tensor_parallel_size: 4
      gpu_memory_utilization: 0.9  # ~61.08GB
      max_cudagraph_capture_size: 0
```

MoE 模型需要更多设备，thinker 使用 4 卡张量并行。设置 `max_cudagraph_capture_size: 0` 禁用 CUDA Graph 捕获。

### 与其他平台配置对比

| 特性 | CUDA | ROCm | NPU | XPU |
|------|------|------|-----|-----|
| enforce_eager (thinker) | false | true | false | false |
| enforce_eager (talker) | false | true | true | false |
| max_cudagraph_capture_size | 默认 | 默认 | 默认 | 0 |
| 注释权重大小 | 无 | 无 | 无 | 有 |

## 总结

XPU 阶段配置在结构上与其他平台一致，但针对 Intel Arc GPU 的显存容量和性能特征进行了参数调整。配置文件中详细注明了各阶段的模型权重大小，为用户的设备选型提供了参考。
