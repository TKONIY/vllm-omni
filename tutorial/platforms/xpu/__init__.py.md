# `__init__.py` -- XPU 包导出

## 文件概述

XPU 子包的初始化文件，导入并导出 `XPUOmniPlatform` 类。

## 关键代码解析

```python
from vllm_omni.platforms.xpu.platform import XPUOmniPlatform

__all__ = ["XPUOmniPlatform"]
```

## 总结

标准包导出文件，使外部可通过 `from vllm_omni.platforms.xpu import XPUOmniPlatform` 引用。
