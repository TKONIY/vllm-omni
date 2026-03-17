# `__init__.py` — 引擎核心数据结构定义

## 文件概述

`__init__.py` 是 `engine/` 模块的入口文件，定义了 vLLM-Omni 引擎层使用的核心数据结构。这些结构体基于 `msgspec.Struct`，用于在多进程 ZMQ 通信中进行高效序列化。它们扩展了 vLLM 原生的 `EngineCoreRequest`、`EngineCoreOutput` 和 `EngineCoreOutputs`，增加了对多模态数据（如预计算嵌入、附加信息字典）的支持。

## 关键代码解析

### 1. PromptEmbedsPayload — 嵌入向量载体

```python
class PromptEmbedsPayload(msgspec.Struct):
    data: bytes
    shape: list[int]
    dtype: str
```

该结构体将 `torch.Tensor` 格式的 prompt 嵌入向量序列化为原始字节流 + 形状 + 数据类型的三元组，便于通过 ZMQ 进行跨进程传输。`data` 存储行主序的原始字节，`shape` 记录 `[seq_len, hidden_size]`，`dtype` 为 `"float16"` 等字符串名称。

### 2. AdditionalInformationEntry — 附加信息条目

```python
class AdditionalInformationEntry(msgspec.Struct):
    # Tensor 形式
    tensor_data: bytes | None = None
    tensor_shape: list[int] | None = None
    tensor_dtype: str | None = None
    # List 形式
    list_data: list[Any] | None = None
```

每个条目支持两种数据形式：
- **张量形式**：与 `PromptEmbedsPayload` 类似，将张量拆为字节、形状和类型
- **列表形式**：直接存储 Python 列表（msgspec 可序列化的基本类型）

两种形式互斥，即 `tensor_data` 和 `list_data` 中有且仅有一个非 None。

### 3. AdditionalInformationPayload — 附加信息字典容器

```python
class AdditionalInformationPayload(msgspec.Struct):
    entries: dict[str, AdditionalInformationEntry]
```

将多个附加信息条目封装为一个字典结构，键为字符串标识（如 `"global_request_id"`、`"ref_code"`），值为上面的 `AdditionalInformationEntry`。

### 4. OmniEngineCoreRequest — 扩展请求类

```python
class OmniEngineCoreRequest(EngineCoreRequest):
    additional_information: AdditionalInformationPayload | None = None
```

继承 vLLM 原生的 `EngineCoreRequest`，新增 `additional_information` 字段。这使得请求可以携带额外的张量或元数据（如说话人参考音频编码），在阶段间传递。

### 5. OmniEngineCoreOutput / OmniEngineCoreOutputs — 扩展输出类

```python
class OmniEngineCoreOutput(EngineCoreOutput):
    pooling_output: dict[str, torch.Tensor] | None = None

class OmniEngineCoreOutputs(EngineCoreOutputs):
    outputs: list[OmniEngineCoreOutput] = []
```

`OmniEngineCoreOutput` 增加了 `pooling_output` 字段，用于承载模型生成的多模态张量输出（如音频隐状态、图像 latent）。`OmniEngineCoreOutputs` 是其列表容器。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `PromptEmbedsPayload` | Struct | 将 prompt 嵌入张量序列化为字节流，用于 ZMQ 传输 |
| `AdditionalInformationEntry` | Struct | 单个附加信息条目，支持张量或列表两种形式 |
| `AdditionalInformationPayload` | Struct | 附加信息字典容器，封装多个条目 |
| `OmniEngineCoreRequest` | Struct | 扩展 vLLM 请求，增加附加信息支持 |
| `OmniEngineCoreOutput` | Struct | 扩展 vLLM 输出，增加多模态张量输出 |
| `OmniEngineCoreOutputs` | Struct | 扩展输出列表容器 |

## 与其他模块的关系

- **`serialization.py`**：`serialize_additional_information()` 函数负责将原始 Python 字典转换为 `AdditionalInformationPayload`
- **`async_omni_engine.py`**：在 `_upgrade_to_omni_request()` 中创建 `OmniEngineCoreRequest`
- **`orchestrator.py`**：`build_engine_core_request_from_tokens()` 构建 `OmniEngineCoreRequest` 用于阶段间转发
- **`output_processor.py`**：处理 `OmniEngineCoreOutput` 中的 `pooling_output` 字段

## 总结

`__init__.py` 定义了 vLLM-Omni 引擎层所有跨进程通信的数据载体。通过 `msgspec.Struct` 实现了高效序列化，同时保持了与 vLLM 原生数据结构的继承兼容性。这些数据结构是整个多阶段推理流水线的通信基础。
