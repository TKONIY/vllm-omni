# `serialization.py` — 附加信息序列化工具

## 文件概述

`serialization.py` 提供了 vLLM-Omni 引擎请求载荷的序列化辅助函数。其核心功能是将 Python 字典形式的 `additional_information`（包含 `torch.Tensor` 和列表等数据）转换为 `AdditionalInformationPayload` 结构体，以便通过 msgspec/ZMQ 进行高效的跨进程传输。

## 关键代码解析

### 1. dtype 名称转换

```python
def dtype_to_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: "float32",
        torch.float: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.int64: "int64",
        torch.long: "int64",
        torch.int32: "int32",
        torch.int16: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))
```

将 `torch.dtype` 转换为稳定的字符串名称。注意同一种类型可能有多个别名（如 `torch.float` 和 `torch.float32`），字典中都做了映射。对于未在映射表中的类型，使用 `str(dtype)` 并移除 `torch.` 前缀作为回退。

### 2. 序列化附加信息

```python
def serialize_additional_information(
    raw_info: dict[str, Any] | AdditionalInformationPayload | None,
    *,
    log_prefix: str | None = None,
) -> AdditionalInformationPayload | None:
    if raw_info is None:
        return None
    if isinstance(raw_info, AdditionalInformationPayload):
        return raw_info  # 已序列化，直接返回

    entries: dict[str, AdditionalInformationEntry] = {}
    for key, value in raw_info.items():
        if isinstance(value, torch.Tensor):
            value_cpu = value.detach().to("cpu").contiguous()
            entries[key] = AdditionalInformationEntry(
                tensor_data=value_cpu.numpy().tobytes(),
                tensor_shape=list(value_cpu.shape),
                tensor_dtype=dtype_to_name(value_cpu.dtype),
            )
            continue

        if isinstance(value, list):
            entries[key] = AdditionalInformationEntry(list_data=value)
            continue

        # 不支持的类型：记录警告并丢弃
        logger.warning(
            "Dropping unsupported additional_information key=%s type=%s",
            key, type(value).__name__,
        )

    return AdditionalInformationPayload(entries=entries) if entries else None
```

序列化逻辑按值类型分流处理：

- **`torch.Tensor`**：先移到 CPU 并确保连续内存布局，然后通过 `numpy().tobytes()` 转为原始字节，同时记录形状和数据类型
- **`list`**：直接存入 `list_data` 字段（msgspec 原生支持序列化）
- **其他类型**：记录警告日志后丢弃，防止不支持的类型导致序列化失败

函数具有幂等性：如果输入已经是 `AdditionalInformationPayload`，直接返回。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `dtype_to_name()` | 函数 | 将 torch.dtype 转换为字符串名称 |
| `serialize_additional_information()` | 函数 | 将原始字典序列化为 AdditionalInformationPayload |

## 与其他模块的关系

- **`__init__.py`**：使用 `AdditionalInformationEntry` 和 `AdditionalInformationPayload` 数据结构
- **`async_omni_engine.py`**：在 `_upgrade_to_omni_request()` 中调用 `serialize_additional_information()`
- **`orchestrator.py`**：在 `build_engine_core_request_from_tokens()` 中调用 `serialize_additional_information()`
- **反序列化**：反序列化在 EngineCore worker 端完成（不在本文件中），通过 numpy.frombuffer + torch.from_numpy 恢复张量

## 总结

`serialization.py` 是一个简洁但关键的工具模块，负责将包含 PyTorch 张量的 Python 字典转换为可通过 ZMQ 传输的 msgspec 结构体。它采用了安全的降级策略（CPU 转移、连续内存、类型过滤），确保序列化过程不会因为意外的数据类型而崩溃。与反序列化端配合，构成了多阶段流水线中跨进程数据传递的基础设施。
