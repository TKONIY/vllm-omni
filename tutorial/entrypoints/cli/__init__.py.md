# `__init__.py` — CLI 包初始化

## 文件概述

CLI 包的初始化文件，导入 benchmark patch 并导出两个核心 CLI 命令类。

## 关键代码解析

```python
from vllm_omni.benchmarks.patch import patch  # 确保 benchmark patch 生效
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand
from .serve import OmniServeCommand

__all__ = ["OmniServeCommand", "OmniBenchmarkServingSubcommand"]
```

`patch` 的导入是为了确保 benchmark 相关的 monkey-patch 在 CLI 初始化时就已生效。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniServeCommand` | 类 (re-export) | serve 子命令 |
| `OmniBenchmarkServingSubcommand` | 类 (re-export) | bench serve 子命令 |

## 总结

标准的包初始化文件，确保 benchmark patch 生效并统一导出 CLI 命令类。
