# hooks/ — Hook 机制子模块

## 模块概述

`hooks/` 子模块提供了非侵入式的模型前向传播拦截机制，是序列并行等高级功能的基础设施。通过在模块上注册 Hook，可以在不修改模型 `forward()` 方法的情况下拦截和修改输入/输出。

## 架构设计

```
ModelHook (Hook 基类)
  ├── SequenceParallelSplitHook (输入分片)
  └── SequenceParallelGatherHook (输出聚合)

HookRegistry (模块级 Hook 管理器)
  ├── register_hook() (注册 Hook)
  ├── dispatch() (调度 forward 调用)
  └── remove_hook() (移除 Hook)

_WrappedForward (forward 代理)
  → 拦截 module.forward() 调用
  → 转发到 HookRegistry.dispatch()
```

## 文件索引

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.md) | 包入口，汇总导出 |
| [`base.py`](base.md) | Hook 基类、HookRegistry、状态管理 |
| [`sequence_parallel.py`](sequence_parallel.md) | 序列并行 Hook 实现（分片/聚合/auto_pad） |

## 核心设计

- **非侵入式**：通过替换 `module.forward` 为 `_WrappedForward` 代理，保存原始 forward 为 `_omni_original_forward`
- **多 Hook 支持**：按名称字母序链式调度 `pre_forward`，逆序调度 `post_forward`
- **序列并行**：支持全序列分片（`SequenceParallelInput`）、部分分片（`SequenceParallelPartialInput`）和自动 padding
- **作用域跟踪**：通过 `ForwardContext._sp_shard_depth` 精确跟踪 SP 的分片/聚合作用域
- **与 diffusers 的对应关系**：SP 对应 diffusers 的 Context Parallelism（CP），`_sp_plan` 对应 `_cp_plan`
