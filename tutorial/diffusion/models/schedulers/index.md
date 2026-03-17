# schedulers/ -- 噪声调度器目录索引

## 目录概述

`schedulers/` 包含扩散模型的自定义噪声调度器实现，为 diffusers 内置调度器的补充。

## 文件列表

| 文件 | 说明 | 文档 |
|------|------|------|
| [`__init__.py`](__init__.md) | 包初始化，导出 FlowUniPCMultistepScheduler |
| [`base.py`](base.md) | 调度器抽象基类 |
| [`scheduling_flow_unipc_multistep.py`](scheduling_flow_unipc_multistep.md) | Flow Matching UniPC 多步调度器 |

## 调度器选择指南

| 调度器 | 适用场景 | 推荐步数 | 特色 |
|--------|---------|---------|------|
| `FlowUniPCMultistepScheduler` | Flow Matching 模型（Wan2.2 等） | 20-30 步 | 预测-校正，高阶精度 |
| `FlowMatchEulerDiscreteScheduler` (diffusers) | FLUX/SD3 等 | 28-50 步 | 简单稳定 |
| `CosineDPMSolverMultistepScheduler` (diffusers) | Stable Audio | 100 步 | 余弦调度 |

## 总结

调度器模块为不同的扩散模型提供适配的采样策略。`FlowUniPCMultistepScheduler` 是本项目的核心自定义调度器，通过 UniPC 预测-校正算法在 Flow Matching 模型上实现更高效的采样。
