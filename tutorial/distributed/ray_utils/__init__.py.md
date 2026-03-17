# `__init__.py` — ray_utils 包入口

## 文件概述

导出三个核心工具函数。

```python
from .utils import calculate_total_bytes, is_ray_initialized, maybe_disable_pin_memory_for_ray
```

## 核心类/函数

| 名称 | 用途 |
|------|------|
| `calculate_total_bytes` | 计算 tensor 分配的总字节数 |
| `is_ray_initialized` | 检查 Ray 是否已初始化 |
| `maybe_disable_pin_memory_for_ray` | Ray 环境下大分配时临时禁用 pin_memory |

## 总结

导入文件，暴露最常用的三个工具函数。
