# worker/ 模块概述

## 模块简介

`worker/` 模块是 vLLM-Omni 推理引擎的**GPU 工作进程层**，负责在 GPU 上实际执行模型推理。该模块扩展了上游 vLLM 的 Worker/ModelRunner 体系，加入了多阶段（multi-stage）流水线、进程级 GPU 显存管理、多模态输出提取等 Omni 特有功能。

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     vLLM-Omni Engine Core                       │
│                   (调度器 / EngineCore)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │ SchedulerOutput
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Worker 层 (本模块)                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐                       │
│  │ OmniWorker   │    │ OmniGPUWorker    │                       │
│  │   Mixin      │    │     Base         │                       │
│  │ (插件加载)    │    │ (显存管理基类)    │                       │
│  └──────┬───────┘    └────────┬─────────┘                       │
│         │  混入               │  继承                            │
│         ▼                     ▼                                  │
│  ┌─────────────────────────────────────┐                        │
│  │         GPUARWorker                 │ ◄── 自回归文本生成阶段   │
│  │  (init_device + GPUARModelRunner)   │                        │
│  └─────────────────┬───────────────────┘                        │
│                    │ 使用                                        │
│                    ▼                                             │
│  ┌─────────────────────────────────────┐                        │
│  │       GPUARModelRunner              │                        │
│  │  (execute_model + sample_tokens)    │                        │
│  │  返回 hidden_states + sampled tokens│                        │
│  └─────────────────┬───────────────────┘                        │
│                    │ 继承                                        │
│                    ▼                                             │
│  ┌─────────────────────────────────────┐                        │
│  │       OmniGPUModelRunner            │ ◄── 公共基础 Runner     │
│  │  (_preprocess / _update_states /    │                        │
│  │   extract_multimodal_outputs /      │                        │
│  │   model_intermediate_buffer 管理)   │                        │
│  └─────────────────────────────────────┘                        │
│                    ▲ 继承                                        │
│  ┌─────────────────────────────────────┐                        │
│  │    GPUGenerationModelRunner         │ ◄── 非自回归生成阶段    │
│  │  (Code2Wav / 扩散模型等)             │                        │
│  │  不做 logits/sampling，直接返回      │                        │
│  │  音频波形等 multimodal_outputs       │                        │
│  └─────────────────┬───────────────────┘                        │
│                    │ 被使用                                      │
│                    ▼                                             │
│  ┌─────────────────────────────────────┐                        │
│  │      GPUGenerationWorker            │                        │
│  │  (init_device + GenerationRunner)   │                        │
│  └─────────────────────────────────────┘                        │
│                                                                 │
│  ┌─────────────────────────────────────┐                        │
│  │       gpu_memory_utils              │ ◄── NVML 显存工具      │
│  │  (per-process GPU memory tracking)  │                        │
│  └─────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## 文件清单

| 文件名 | 用途 |
|--------|------|
| `__init__.py` | 包初始化（空文件） |
| `base.py` | GPU Worker 基类，提供进程级显存管理 |
| `gpu_model_runner.py` | Omni 公共 ModelRunner 基类，扩展上游 GPUModelRunner |
| `gpu_ar_model_runner.py` | 自回归（AR）模型运行器，返回 hidden states + sampled tokens |
| `gpu_ar_worker.py` | 自回归 Worker，初始化设备并创建 GPUARModelRunner |
| `gpu_generation_model_runner.py` | 非自回归生成模型运行器（如 Code2Wav） |
| `gpu_generation_worker.py` | 非自回归 Worker，初始化设备并创建 GPUGenerationModelRunner |
| `gpu_memory_utils.py` | 基于 NVML 的进程级 GPU 显存查询工具 |
| `mixins.py` | OmniWorkerMixin，确保 Worker 进程加载 Omni 插件 |

## 核心设计理念

1. **多阶段流水线**：vLLM-Omni 将推理拆分为多个阶段（如"思考"阶段用 AR 模型、"发声"阶段用非 AR 生成模型），每个阶段有独立的 Worker + ModelRunner 对。

2. **进程级显存管理**：多个阶段可能共享同一张 GPU，传统的全局显存统计会导致冲突。`base.py` 和 `gpu_memory_utils.py` 利用 NVML 的 per-PID 显存追踪解决此问题。

3. **中间缓冲区（model_intermediate_buffer）**：跨阶段传递请求级别的中间数据（如 hidden states、音频编码等），由 `OmniGPUModelRunner` 统一管理。

4. **兼容上游 vLLM**：所有类均继承自 vLLM v1 的对应基类（`GPUWorker`、`GPUModelRunner`），保留了 CUDA Graph、投机解码、TP/PP/DP 等高级特性。
