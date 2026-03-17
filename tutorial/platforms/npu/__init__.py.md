# `__init__.py` -- NPU 包导出

## 文件概述

NPU 子包的初始化文件，导入并导出 `NPUOmniPlatform` 类。

## 关键代码解析

```python
from vllm_omni.platforms.npu.platform import NPUOmniPlatform

__all__ = ["NPUOmniPlatform"]
```

## 总结

标准包导出文件，使外部可通过 `from vllm_omni.platforms.npu import NPUOmniPlatform` 引用。
