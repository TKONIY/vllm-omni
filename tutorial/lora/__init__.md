# `__init__.py` — LoRARequest 导出

## 文件概述

该文件从 vLLM 重导出 `LoRARequest` 类，使用户可以通过 `vllm_omni.lora` 直接访问，无需关心底层来源。

## 关键代码解析

```python
from vllm.lora.request import LoRARequest

__all__ = ["LoRARequest"]
```

直接重导出 vLLM 的 `LoRARequest`，提供统一的用户接口。

## 总结

简单的重导出文件，将 vLLM 的 LoRA 请求类暴露为 vllm-omni 的公开 API，方便用户使用。
