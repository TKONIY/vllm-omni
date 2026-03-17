# `logging.py` — 日志工具

## 文件概述

该文件提供统一的日志获取函数 `get_connector_logger`，优先使用 vLLM 的日志系统，回退到标准 `logging` 模块。

## 关键代码解析

```python
try:
    from vllm.logger import init_logger as _vllm_init_logger
except Exception:
    _vllm_init_logger = None

def get_connector_logger(name: str) -> logging.Logger:
    """Return a logger preferring vLLM's init_logger when available."""
    return _vllm_init_logger(name) if _vllm_init_logger else logging.getLogger(name)
```

设计意图：
- 当 vLLM 可用时，使用 vLLM 的日志格式和配置（统一日志风格）
- 当独立运行（如单元测试）时，回退到标准 `logging`

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `get_connector_logger()` | function | 获取兼容 vLLM 的 logger |

## 与其他模块的关系

- 被几乎所有 `omni_connectors` 内的模块使用
- 依赖 vLLM 的 `init_logger`（可选）

## 总结

简洁的日志工具函数，提供 vLLM 日志系统的优雅降级。
