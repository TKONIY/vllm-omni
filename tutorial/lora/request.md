# `request.py` — LoRARequest 重导出

## 文件概述

`request.py` 的功能与 `__init__.py` 相同，从 vLLM 重导出 `LoRARequest` 类。该文件作为用户面向的变量定义存在，确保用户可以直接从 `vllm_omni` 导入 `LoRARequest`。

## 关键代码解析

```python
from vllm.lora.request import LoRARequest

__all__ = ["LoRARequest"]
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `LoRARequest` | 类（重导出） | vLLM 的 LoRA 请求类 |

## 与其他模块的关系

- **来源**: `vllm.lora.request.LoRARequest`
- **使用场景**: 在 vllm-omni 的 API 层和基准测试中用于指定 LoRA 适配器

## 总结

纯重导出文件，无额外逻辑。`LoRARequest` 的实际实现在 vLLM 中。
