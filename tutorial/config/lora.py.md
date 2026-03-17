# `config/lora.py` — LoRA 配置

## 文件概述

`lora.py` 将 vLLM 的 `LoRAConfig` 直接转发导出，使用户可以从 `vllm_omni.config` 直接导入 LoRA 配置，无需了解底层来源。

## 关键代码解析

```python
from vllm.config.lora import LoRAConfig

__all__ = ["LoRAConfig"]
```

当前直接复用 vLLM 的 LoRA 实现。未来如有 omni 特有的 LoRA 扩展，可在此文件中添加。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `LoRAConfig` | 类（转发） | LoRA 适配器配置 |

## 总结

纯转发模块，提供统一的 LoRA 配置导入路径。
