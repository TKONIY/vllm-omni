# `__init__.py` -- CUDA 包导出

## 文件概述

CUDA 子包的初始化文件，仅负责从 `platform.py` 导入并导出 `CudaOmniPlatform` 类。

## 关键代码解析

```python
from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

__all__ = ["CudaOmniPlatform"]
```

## 总结

标准的 Python 包导出文件，使外部模块可通过 `from vllm_omni.platforms.cuda import CudaOmniPlatform` 直接引用平台类。
