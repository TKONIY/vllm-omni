# `core/sched/output.py` — 调度器输出数据结构

## 文件概述

`output.py` 定义了调度器的输出数据结构，扩展 vLLM 的 `NewRequestData`、`CachedRequestData` 和 `SchedulerOutput`，添加 omni 特有的字段以支持嵌入传递、附加信息和 KV 缓存传输元数据。

## 关键代码解析

### OmniNewRequestData

```python
@dataclass
class OmniNewRequestData(NewRequestData):
    external_req_id: str | None = None
    additional_information: AdditionalInformationPayload | None = None
```

扩展新请求数据，增加：
- `external_req_id`：外部请求 ID，用于跨阶段追踪
- `additional_information`：附加信息负载，包含跨阶段传递的张量或列表

提供工厂方法从 `Request` 对象创建：

```python
@classmethod
def from_request(cls, request, block_ids, prefill_token_ids=None):
    return cls(
        req_id=request.request_id,
        external_req_id=getattr(request, "external_req_id", None),
        prompt_embeds=getattr(request, "prompt_embeds", None),
        additional_information=getattr(request, "additional_information", None),
        # ... 基类字段
    )
```

### OmniCachedRequestData

```python
@dataclass
class OmniCachedRequestData(CachedRequestData):
    prompt_token_ids: dict[str, list[int]]
    additional_information: dict[str, dict | None]
```

扩展缓存请求数据，增加：
- `prompt_token_ids`：请求 ID 到 prompt token 列表的映射
- `additional_information`：请求 ID 到附加信息的映射

主要用于生成调度器的连续推理步中，传递完整的 prompt 信息。

### OmniSchedulerOutput

```python
@dataclass
class OmniSchedulerOutput(SchedulerOutput):
    finished_requests_needing_kv_transfer: dict[str, dict] = field(default_factory=dict)
```

扩展调度器输出，增加 KV 缓存传输元数据字段。结构为：

```python
{
    "request_id": {
        "seq_len": int,      # 序列长度
        "block_ids": list[int]  # KV 缓存块 ID 列表
    }
}
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniNewRequestData` | 数据类 | 新请求调度数据（含嵌入和附加信息） |
| `OmniCachedRequestData` | 数据类 | 缓存请求数据（含 prompt token） |
| `OmniSchedulerOutput` | 数据类 | 调度器输出（含 KV 传输元数据） |
| `from_request` | 类方法 | 从 Request 创建 OmniNewRequestData |

## 与其他模块的关系

- 继承 `vllm.v1.core.sched.output` 中的基类
- 被 `omni_ar_scheduler.py` 和 `omni_generation_scheduler.py` 使用
- 引用 `engine.AdditionalInformationPayload` 进行数据传输
- 传递给模型执行器（model runner）用于推理

## 总结

`output.py` 定义了调度器与模型执行器之间的数据接口，通过扩展 vLLM 的基础数据结构，实现了多阶段流水线中嵌入传递、附加信息传输和 KV 缓存跨阶段迁移等核心功能。
