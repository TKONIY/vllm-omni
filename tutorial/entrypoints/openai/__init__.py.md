# `__init__.py` — OpenAI 模块入口

## 文件概述

OpenAI 兼容 API 包的入口文件，导出核心服务器函数和服务类。

## 关键代码解析

```python
from vllm_omni.entrypoints.openai.api_server import (
    build_async_omni, omni_init_app_state, omni_run_server,
)
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

__all__ = [
    "omni_run_server", "build_async_omni", "omni_init_app_state",
    "OmniOpenAIServingChat",
]
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `omni_run_server` | 函数 | 服务器启动入口 |
| `build_async_omni` | 函数 | 构建 AsyncOmni 引擎 |
| `omni_init_app_state` | 函数 | 初始化 FastAPI 应用状态 |
| `OmniOpenAIServingChat` | 类 | 聊天补全处理器 |

## 总结

标准的包初始化文件，导出 API 服务器的核心公共接口。
