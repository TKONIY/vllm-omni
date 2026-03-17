# `request.py` — 扩展请求类

## 文件概述

`request.py` 定义了 `OmniRequest` 类，扩展 vLLM 的 `Request` 以支持跨阶段的 prompt 嵌入传递和附加信息负载。该类通过 `patch.py` 替换 vLLM 原始的 `Request`，在整个推理管线中透明使用。

## 关键代码解析

### 构造函数扩展

```python
class OmniRequest(Request):
    def __init__(
        self,
        prompt_embeds: PromptEmbedsPayload | torch.Tensor | None = None,
        external_req_id: str | None = None,
        additional_information: AdditionalInformationPayload | None = None,
        *args,
        **kwargs,
    ):
        prompt_embeds_tensor = self._maybe_decode_prompt_embeds(prompt_embeds)
        super().__init__(prompt_embeds=prompt_embeds_tensor, *args, **kwargs)
        self.prompt_embeds_payload = (
            prompt_embeds if isinstance(prompt_embeds, PromptEmbedsPayload) else None
        )
        self.external_req_id = external_req_id
        self.additional_information = additional_information
```

新增三个关键字段：
- `prompt_embeds`：支持 `PromptEmbedsPayload`（序列化格式）或 `torch.Tensor`（直接张量）
- `external_req_id`：外部请求 ID，用于跨阶段追踪
- `additional_information`：附加信息负载（张量或列表），用于阶段间数据传递

### 序列化嵌入解码

```python
@staticmethod
def _maybe_decode_prompt_embeds(prompt_embeds):
    if isinstance(prompt_embeds, PromptEmbedsPayload):
        dtype = getattr(np, prompt_embeds.dtype)
        arr = np.frombuffer(prompt_embeds.data, dtype=dtype)
        arr = arr.reshape(prompt_embeds.shape)
        return torch.from_numpy(arr)
    return prompt_embeds
```

当嵌入以 `PromptEmbedsPayload`（序列化字节）形式传入时，自动解码为 `torch.Tensor`。这支持跨进程/跨节点的嵌入传输场景。

### 工厂方法

```python
@classmethod
def from_engine_core_request(cls, request, block_hasher):
    return cls(
        request_id=request.request_id,
        external_req_id=request.external_req_id,
        prompt_embeds=request.prompt_embeds,
        additional_information=request.additional_information,
        # ... 其他标准字段
    )
```

从 `OmniEngineCoreRequest` 创建 `OmniRequest`，在引擎核心层完成请求对象的转换。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniRequest` | 类 | 扩展的请求类，支持嵌入和附加信息 |
| `_maybe_decode_prompt_embeds` | 静态方法 | 将序列化嵌入解码为 Tensor |
| `from_engine_core_request` | 类方法 | 从引擎核心请求创建 OmniRequest |

## 与其他模块的关系

- 继承 `vllm.v1.request.Request`
- 被 `patch.py` 替换到所有 vLLM 模块中
- 引用 `engine.PromptEmbedsPayload` 和 `AdditionalInformationPayload` 用于数据传输
- 被 `core/sched/` 调度器管理和调度

## 总结

`OmniRequest` 是 vllm-omni 请求系统的核心扩展，通过增加嵌入传递和附加信息支持，实现了多阶段流水线中数据的无缝流转。序列化/反序列化机制使其支持分布式部署场景。
