# `serve.py` — 基准测试主入口

## 文件概述

`serve.py` 是 benchmarks 模块的顶层入口文件，提供同步的 `main` 函数，内部调用 vLLM 原生的 `main_async` 来执行基准测试。该文件本身非常简洁，其核心作用是将异步入口封装为同步调用。

## 关键代码解析

```python
import argparse
import asyncio
from typing import Any

from vllm.benchmarks.serve import main_async


def main(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(main_async(args))
```

整个文件仅包含一个 `main` 函数：
- 接收 `argparse.Namespace` 参数对象
- 通过 `asyncio.run()` 同步运行 vLLM 的异步基准测试入口 `main_async`
- 返回包含基准测试结果的字典

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `main(args)` | 函数 | 同步入口，封装 `main_async` 的调用 |

## 与其他模块的关系

- **依赖 vLLM**: 直接调用 `vllm.benchmarks.serve.main_async`。
- **与 patch 模块配合**: 需要先导入 `patch/patch.py` 使猴子补丁生效，`main_async` 内部才会使用 vllm-omni 自定义的后端和指标。

## 总结

`serve.py` 是一个极简的入口封装，将 vLLM 的异步基准测试流程暴露为同步 API。实际的多模态扩展逻辑在 `patch/patch.py` 中通过猴子补丁注入。
