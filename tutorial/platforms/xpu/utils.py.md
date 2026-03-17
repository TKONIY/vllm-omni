# `utils.py` -- XPU 工具函数

## 文件概述

`xpu/utils.py` 提供了 `torch_cuda_wrapper` 上下文管理器，用于将 `torch.cuda` 的 Stream 相关 API 替换为 XPU 对应的 API。这是 XPU 平台能够复用 GPU 通用代码的关键机制。

## 关键代码解析

```python
from contextlib import contextmanager
import torch

@contextmanager
def torch_cuda_wrapper():
    try:
        torch.cuda.Stream = torch.xpu.Stream
        torch.cuda.default_stream = torch.xpu.current_stream
        torch.cuda.current_stream = torch.xpu.current_stream
        torch.cuda.stream = torch.xpu.stream
        yield
    finally:
        pass
```

该上下文管理器在进入时将以下 `torch.cuda` API 替换为 `torch.xpu` 等效实现：

| torch.cuda 原始 API | 替换为 torch.xpu API |
|---------------------|---------------------|
| `torch.cuda.Stream` | `torch.xpu.Stream` |
| `torch.cuda.default_stream` | `torch.xpu.current_stream` |
| `torch.cuda.current_stream` | `torch.xpu.current_stream` |
| `torch.cuda.stream` | `torch.xpu.stream` |

注意 `finally` 块为空，意味着替换是永久性的（不恢复原始 API）。这在 XPU 环境中是安全的，因为不存在 CUDA 设备。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `torch_cuda_wrapper()` | 上下文管理器 | 将 CUDA Stream API 替换为 XPU 等效 API |

## 与其他模块的关系

- **使用者**：`XPUARModelRunner.__init__()` 和 `XPUGenerationModelRunner.__init__()`
- **目的**：使 GPU 通用代码中的 CUDA Stream 操作能在 XPU 设备上正常工作

## 总结

`torch_cuda_wrapper` 是一个简单但关键的兼容层，通过猴子补丁（monkey patching）方式使依赖 `torch.cuda` Stream API 的代码能透明地运行在 XPU 设备上。这种设计避免了大量代码复制，使 XPU 平台能以最小改动复用 GPU 通用实现。
