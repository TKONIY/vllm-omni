# diffusion/ — 扩散模型推理核心模块

## 模块概述

`diffusion/` 是 vLLM-Omni 中扩散模型推理的核心模块，提供了完整的扩散模型加载、分布式执行、缓存加速、性能分析等基础设施。该模块支持 20+ 种扩散模型（包括图像生成、视频生成、音频生成和图像编辑），并支持多种并行策略（数据并行、张量并行、序列并行、流水线并行等）。

## 架构概览

```
DiffusionEngine (顶层入口)
  ├── Registry (模型注册与前后处理)
  ├── Executor (执行器抽象)
  │     └── MultiprocDiffusionExecutor (多进程执行器)
  │           ├── Scheduler (请求调度与消息路由)
  │           └── WorkerProc (进程封装)
  │                 └── DiffusionWorker (GPU Worker)
  │                       └── DiffusionModelRunner (模型加载与执行)
  ├── ForwardContext (前向上下文管理)
  ├── Hooks (非侵入式模型修改)
  │     ├── ModelHook / HookRegistry (基础框架)
  │     └── SequenceParallelSplitHook / GatherHook (序列并行)
  ├── Layers (多平台算子)
  │     ├── CustomOp (平台调度基类)
  │     ├── AdaLayerNorm (自适应归一化)
  │     └── RotaryEmbedding (旋转位置编码)
  └── IPC (共享内存张量传输)
```

## 文件索引

### 根文件

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.md) | 包入口 |
| [`data.py`](data.md) | 核心数据结构与配置（OmniDiffusionConfig 等） |
| [`diffusion_engine.py`](diffusion_engine.md) | 扩散推理引擎顶层入口 |
| [`registry.py`](registry.md) | 模型注册表与初始化 |
| [`request.py`](request.md) | 扩散请求数据结构 |
| [`scheduler.py`](scheduler.md) | 请求调度器（MessageQueue 通信） |
| [`forward_context.py`](forward_context.md) | 前向传播上下文管理 |
| [`compile.py`](compile.md) | 区域编译（torch.compile）加速 |
| [`envs.py`](envs.md) | 环境变量与包检测 |
| [`ipc.py`](ipc.md) | POSIX 共享内存张量传输 |
| [`stage_diffusion_client.py`](stage_diffusion_client.md) | 多阶段架构的扩散客户端 |

### 子模块

| 子模块 | 说明 | 索引 |
|--------|------|------|
| [`executor/`](executor/index.md) | 执行器抽象与多进程实现 |
| [`hooks/`](hooks/index.md) | Hook 机制与序列并行 |
| [`layers/`](layers/index.md) | 多平台自定义算子层 |
| [`utils/`](utils/index.md) | 工具函数（HF 检测、网络、配置） |
| [`worker/`](worker/index.md) | GPU Worker 与模型运行器 |
| [`profiler/`](profiler/index.md) | 性能分析器 |

## 核心数据流

```
用户请求 → DiffusionEngine.step()
  → pre_process_func() (前处理)
  → Executor.add_req()
    → Scheduler.add_req()
      → MessageQueue.enqueue() (广播到所有 Worker)
        → WorkerProc.worker_busy_loop()
          → DiffusionWorker.execute_model()
            → DiffusionModelRunner.execute_model()
              → pipeline.forward() (实际推理)
          → pack_diffusion_output_shm() (SHM 打包)
      → ResultQueue.dequeue() (接收结果)
      → unpack_diffusion_output_shm() (SHM 解包)
  → post_process_func() (后处理)
  → OmniRequestOutput (最终输出)
```

## 关键设计特点

1. **多进程分布式架构**：通过 MessageQueue + SHM 实现高效的 Worker 间通信
2. **非侵入式并行**：通过 Hook 机制实现序列并行，无需修改模型代码
3. **多平台支持**：通过 CustomOp 基类统一 CUDA/ROCm/NPU/XPU 的算子调度
4. **懒加载模型注册**：通过 `_ModelRegistry` 实现按需加载，减少启动开销
5. **可扩展执行器**：抽象基类支持多种分布式后端（多进程、Ray 等）
