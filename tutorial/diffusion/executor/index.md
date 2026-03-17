# executor/ — 执行器子模块

## 模块概述

`executor/` 子模块定义了扩散模型的执行器架构，负责管理 Worker 进程的生命周期、请求调度和分布式 RPC 通信。

## 架构设计

```
DiffusionExecutor (抽象基类)
  └── MultiprocDiffusionExecutor (多进程实现)
        ├── Scheduler (请求调度)
        ├── WorkerProc × N (GPU Worker 进程)
        └── BackgroundResources (资源清理)
```

## 文件索引

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.md) | 包入口 |
| [`abstract.py`](abstract.md) | 执行器抽象基类，含工厂方法 |
| [`multiproc_executor.py`](multiproc_executor.md) | 多进程执行器实现 |

## 核心设计

- **抽象基类**：`DiffusionExecutor` 定义了 `add_req`、`collective_rpc`、`check_health`、`shutdown` 四个核心接口
- **工厂方法**：`get_class` 根据 `distributed_executor_backend` 配置动态选择执行器后端
- **资源管理**：通过 `weakref.finalize` 和 `BackgroundResources` 保证 Worker 进程的可靠清理
- **通信机制**：基于 vLLM 的 `MessageQueue`（共享内存）进行高效的请求广播和结果收集
