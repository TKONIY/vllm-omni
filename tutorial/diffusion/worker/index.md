# worker/ — GPU Worker 子模块

## 模块概述

`worker/` 子模块实现了扩散模型的 GPU Worker 架构，遵循 Runner-Worker 分离模式：Worker 负责基础设施（设备、分布式环境），Runner 负责模型操作（加载、编译、执行）。

## 架构设计

```
WorkerProc (进程封装)
  └── WorkerWrapperBase (扩展包装器)
        └── DiffusionWorker (GPU Worker)
              ├── init_device() (设备与分布式初始化)
              ├── sleep()/wake_up() (休眠/唤醒)
              ├── LoRA 管理
              └── DiffusionModelRunner (模型运行器)
                    ├── load_model() (加载/编译/缓存)
                    └── execute_model() (推理执行)
```

## 文件索引

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.md) | 包入口，导出核心类 |
| [`diffusion_model_runner.py`](diffusion_model_runner.md) | 模型运行器（加载/编译/执行） |
| [`diffusion_worker.py`](diffusion_worker.md) | GPU Worker、进程封装和扩展包装器 |

## 核心设计

- **Runner-Worker 分离**：Worker 管理 GPU 设备和分布式环境，Runner 管理模型生命周期
- **动态扩展**：`WorkerWrapperBase` 通过 Python `type()` 实现运行时 mixin 继承
- **休眠模式**：支持 Level 1（卸载权重）和 Level 2（额外保存 buffer）的内存管理
- **消息循环**：`WorkerProc` 通过 `MessageQueue` 接收请求，支持 RPC、生成和关闭三种消息类型
- **HSDP 兼容**：推理时使用 `torch.no_grad()` 而非 `torch.inference_mode()` 以兼容 HSDP2
