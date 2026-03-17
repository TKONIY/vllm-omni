# `output_processor.py` — 多模态输出处理器

## 文件概述

`output_processor.py` 实现了 vLLM-Omni 的多模态输出处理系统，包含两个核心类：`OmniRequestState`（扩展的请求状态）和 `MultimodalOutputProcessor`（扩展的输出处理器）。该模块解决了 vLLM 原生输出处理器仅支持文本输出的限制，增加了对音频、图像、latent 等多模态张量的累积、合并和输出能力。

## 关键代码解析

### 1. OmniRequestState — 多模态请求状态

```python
class OmniRequestState(RequestState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mm_type: str | None = None          # 多模态类型 (audio/image/latent)
        self.mm_accumulated: dict[str, Any] | None = None  # 累积的多模态数据
```

在 vLLM 的 `RequestState` 基础上增加了两个字段：`mm_type` 标识当前请求的多模态类型，`mm_accumulated` 存储逐步累积的多模态张量数据。

### 2. 多模态张量累积

```python
def add_multimodal_tensor(self, payload: Any | None, mm_type: str | None) -> None:
    if payload is None:
        return

    # 将张量移到 CPU
    def _to_cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu", non_blocking=True).contiguous()
        return x

    # 规范化输入
    if isinstance(payload, dict):
        incoming: dict[str, Any] = {}
        target_key = self.mm_type or "hidden"
        for k, v in payload.items():
            # 统一键名：model_outputs -> target_key, hidden -> target_key
            if k == "model_outputs":
                k = target_key
            elif k == "hidden" and target_key != "hidden":
                k = target_key
            incoming[k] = _to_cpu(v) if not isinstance(v, dict) else {
                str(sk): _to_cpu(sv) for sk, sv in v.items()
            }

    # 合并到累积字典
    if self.mm_accumulated is None:
        self.mm_accumulated = incoming
    else:
        for k, v in incoming.items():
            existing = self.mm_accumulated.get(k)
            if existing is None:
                self.mm_accumulated[k] = v
            elif isinstance(v, torch.Tensor) and isinstance(existing, torch.Tensor):
                self.mm_accumulated[k] = [existing, v]  # 使用列表避免 O(n^2) 重复 cat
            elif isinstance(v, torch.Tensor) and isinstance(existing, list):
                existing.append(v)
            # ...
```

核心累积逻辑的设计要点：
- **CPU 转移**：所有张量立即移到 CPU，避免 GPU 内存泄漏
- **键名统一**：不同 runner 产出的键名不同（`hidden`、`model_outputs`），统一映射为语义键（如 `audio`、`latent`）
- **O(n) 累积**：使用列表暂存张量，避免每次都做 `torch.cat`（O(n^2) 开销）

### 3. 张量合并（延迟执行）

```python
def _consolidate_multimodal_tensors(self) -> None:
    if self.mm_accumulated is None:
        return
    for k, v in self.mm_accumulated.items():
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            if k == "audio":
                continue  # 音频维度可能不一致，跳过合并
            elif k == "sr":
                self.mm_accumulated[k] = v[-1]  # 采样率取最后值
            else:
                self.mm_accumulated[k] = torch.cat(v, dim=0)
```

合并策略：
- **通用张量**：沿 dim=0 拼接
- **音频**：跳过拼接（shape 可能不一致，由下游处理）
- **采样率**：取最后一个值（标量常量）
- **失败回退**：拼接失败时保留最后一个张量

### 4. 输出生成 — make_request_output

```python
def make_request_output(self, new_token_ids, pooling_output, finish_reason, stop_reason, ...):
    # 纯 pooling 请求走基类逻辑
    if self.detokenizer is None and pooling_output is not None:
        return super().make_request_output(...)

    finished = finish_reason is not None
    final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

    if not finished and final_only:
        return None

    # 完成时合并张量
    if finished:
        self._consolidate_multimodal_tensors()

    # 创建输出
    output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)
    return self._new_request_output(external_req_id, [output], finished)
```

关键设计：即使请求同时产出文本和多模态数据，也始终走文本输出路径（`CompletionOutput`），而非 pooling 输出路径。多模态数据作为附加属性挂载在 `CompletionOutput` 上。

### 5. 多模态数据附加

```python
def _new_completion_output(self, token_ids, finish_reason, stop_reason, routed_experts=None):
    base_output = super()._new_completion_output(token_ids, finish_reason, stop_reason, routed_experts)
    if self.mm_accumulated is not None:
        if not hasattr(base_output, "multimodal_output"):
            setattr(base_output, "multimodal_output", {})
        mm_out = getattr(base_output, "multimodal_output")
        if isinstance(mm_out, dict):
            for k, v in self.mm_accumulated.items():
                mm_out[k] = v
        else:
            setattr(base_output, "multimodal_output", self.mm_accumulated)
    return base_output
```

将累积的多模态数据字典挂载到 `CompletionOutput.multimodal_output` 属性上，供下游消费者（如 API 端点）读取。

### 6. MultimodalOutputProcessor — 拦截与委托

```python
class MultimodalOutputProcessor(VLLMOutputProcessor):
    def __init__(self, tokenizer, *, log_stats, engine_core_output_type=None, ...):
        super().__init__(tokenizer=tokenizer, log_stats=log_stats, ...)
        self.engine_core_output_type = engine_core_output_type

    def process_outputs(self, engine_core_outputs, ...):
        for eco in engine_core_outputs:
            req_state = self.request_states.get(eco.request_id)
            if req_state is None or not isinstance(req_state, OmniRequestState):
                continue
            if eco.pooling_output is not None and req_state.detokenizer is not None:
                mm_type = getattr(eco, "output_type", self.engine_core_output_type) or ""
                req_state.add_multimodal_tensor(eco.pooling_output, mm_type.lower())
                eco.pooling_output = None  # 强制走文本路径
        return super().process_outputs(engine_core_outputs, ...)
```

处理流程：
1. **拦截**：在调用基类处理之前，从 `EngineCoreOutput.pooling_output` 中捕获多模态张量
2. **累积**：将张量通过 `add_multimodal_tensor()` 累积到请求状态中
3. **清除**：将 `pooling_output` 置 None，强制基类走文本 detokenization 路径
4. **委托**：调用基类 `process_outputs()` 处理文本输出

### 7. 请求注册

```python
def add_request(self, request, prompt, parent_req=None, request_index=0, queue=None):
    request_id = request.request_id
    req_state = self.request_states.get(request_id)
    if req_state is not None:
        self._update_streaming_request_state(req_state, request, prompt)
        return

    req_state = OmniRequestState.from_new_request(
        tokenizer=self.tokenizer,
        request=request,
        prompt=prompt,
        parent_req=parent_req,
        # ...
    )
    self.request_states[request_id] = req_state
```

注册新请求时创建 `OmniRequestState`（而非基类的 `RequestState`）。如果请求已存在（如流式请求），则更新现有状态。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniRequestState` | 类 | 扩展请求状态，支持多模态张量累积 |
| `OmniRequestState.add_multimodal_tensor()` | 方法 | 增量累积多模态张量数据 |
| `OmniRequestState._consolidate_multimodal_tensors()` | 方法 | 请求完成时合并累积的张量列表 |
| `OmniRequestState.make_request_output()` | 方法 | 生成包含多模态数据的请求输出 |
| `OmniRequestState._new_completion_output()` | 方法 | 在 CompletionOutput 上附加多模态数据 |
| `MultimodalOutputProcessor` | 类 | 多模态输出处理器，拦截 pooling_output 并委托文本处理给基类 |
| `MultimodalOutputProcessor.add_request()` | 方法 | 注册请求（使用 OmniRequestState） |
| `MultimodalOutputProcessor.process_outputs()` | 方法 | 拦截多模态数据后委托基类处理文本 |

## 与其他模块的关系

- **`orchestrator.py`**：Orchestrator 为每个阶段持有一个 `MultimodalOutputProcessor`，调用其 `process_outputs()`
- **`async_omni_engine.py`**：Stage-0 的 `MultimodalOutputProcessor` 在 `_attach_llm_stage()` 中创建
- **`__init__.py`**：处理 `OmniEngineCoreOutput` 中的 `pooling_output` 字段
- **`vllm_omni/outputs.py`**：输出为 `OmniRequestOutput` 类型
- **vLLM 基类**：继承 `RequestState` 和 `OutputProcessor`，复用文本 detokenization 逻辑

## 总结

`output_processor.py` 巧妙地在 vLLM 的纯文本输出处理管道中"嫁接"了多模态支持。通过拦截-累积-合并-附加的四步策略，在不修改基类逻辑的前提下实现了音频、图像等张量数据的增量收集和输出。O(n) 的列表累积策略和延迟合并设计确保了在长序列生成场景下的性能表现。
