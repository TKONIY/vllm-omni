# `__init__.py` — worker 包入口

## 文件概述

`worker/__init__.py` 汇总导出了 worker 子包的核心类：`DiffusionModelRunner`、`DiffusionWorker` 和 `WorkerProc`。

## 关键代码解析

```python
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerProc

__all__ = ["DiffusionModelRunner", "DiffusionWorker", "WorkerProc"]
```

## 核心类/函数

| 名称 | 来源 | 说明 |
|------|------|------|
| `DiffusionModelRunner` | `diffusion_model_runner.py` | 模型加载与推理执行 |
| `DiffusionWorker` | `diffusion_worker.py` | GPU Worker，管理设备和分布式环境 |
| `WorkerProc` | `diffusion_worker.py` | 进程封装，运行 Worker 主循环 |

## 总结

该文件作为 worker 子包的公共 API 入口，导出了三个核心类供外部模块使用。
