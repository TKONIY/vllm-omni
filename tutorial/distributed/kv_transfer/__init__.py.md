# `__init__.py` — kv_transfer 模块说明

## 文件概述

该文件仅包含模块级别的 docstring，说明 `kv_transfer` 包的用途：为 PD 分离场景提供 monkey-patch 版本的 vLLM KV transfer 连接器，修复 request-ID 不匹配问题。

## 关键代码解析

```python
"""Patched KV transfer connectors for PD disaggregation.

This package provides monkey-patched versions of vLLM's native KV transfer
connectors (e.g. MooncakeConnector) that fix the request-ID mismatch problem
in prefill-decode disaggregation.
"""
```

没有任何导入或导出——实际逻辑全部在 `monkey_patch.py` 中。

## 总结

纯文档文件，定义了包的用途说明。
