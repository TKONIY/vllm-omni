# lora 模块索引

本模块为 vllm-omni 提供 LoRA（Low-Rank Adaptation）适配器支持，包括请求定义和工具函数。

## 模块结构

```
lora/
├── __init__.py    # 导出 LoRARequest
├── request.py     # 重导出 vLLM 的 LoRARequest（已合并到 __init__.py）
└── utils.py       # LoRA 工具函数（稳定 ID 生成）
```

## 文档列表

| 文件 | 说明 |
|------|------|
| [__init__.md](__init__.md) | 包初始化与 LoRARequest 导出 |
| [request.md](request.md) | LoRARequest 重导出 |
| [utils.md](utils.md) | 稳定 LoRA ID 生成工具 |

## 模块间关系

- `__init__.py` 从 vLLM 重导出 `LoRARequest`，使用户可以直接从 `vllm_omni.lora` 导入。
- `utils.py` 提供 `stable_lora_int_id` 函数，为 LoRA 适配器生成跨进程稳定的整数 ID。
