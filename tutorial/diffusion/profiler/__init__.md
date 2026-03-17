# `__init__.py` — profiler 包入口

## 文件概述

`profiler/__init__.py` 导出了性能分析器的核心接口，并设置默认的 profiler 实现为 `TorchProfiler`。

## 关键代码解析

```python
from .torch_profiler import TorchProfiler

# Default profiler - can be changed later via config
CurrentProfiler = TorchProfiler

__all__ = ["CurrentProfiler", "TorchProfiler"]
```

`CurrentProfiler` 是一个模块级变量，指向当前使用的 profiler 类。默认为 `TorchProfiler`，未来可通过配置切换到其他实现。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `CurrentProfiler` | 类引用 | 当前激活的 profiler 类，默认为 TorchProfiler |
| `TorchProfiler` | 类 | 基于 torch.profiler 的实现 |

## 与其他模块的关系

- 被 `worker/diffusion_worker.py` 的 `DiffusionWorker.start_profile` 和 `stop_profile` 使用。

## 总结

该文件作为 profiler 子包的入口，通过 `CurrentProfiler` 变量提供了可切换的 profiler 机制。
