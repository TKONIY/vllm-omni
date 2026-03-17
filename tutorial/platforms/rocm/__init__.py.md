# `__init__.py` -- ROCm 包导出

## 文件概述

ROCm 子包的初始化文件，导入并导出 `RocmOmniPlatform` 类。

## 关键代码解析

```python
from vllm_omni.platforms.rocm.platform import RocmOmniPlatform

__all__ = ["RocmOmniPlatform"]
```

## 总结

标准包导出文件，使外部可通过 `from vllm_omni.platforms.rocm import RocmOmniPlatform` 引用。
