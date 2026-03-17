# `npu_ar_model_runner.py` -- NPU 自回归 ModelRunner

## 文件概述

`npu_ar_model_runner.py` 实现了 `NPUARModelRunner`，用于 Omni 模型的自回归（Autoregressive）推理阶段（thinker/talker）。这是 NPU 平台代码量最大的文件，包含完整的 `execute_model` 和 `sample_tokens` 实现，处理模型前向传播、采样、KV 缓存传输和多模态输出等复杂逻辑。

## 关键代码解析

### 1. ExecuteModelState

```python
class ExecuteModelState(NamedTuple):
    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: PerLayerAttnMetadata
    positions: torch.Tensor
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    multimodal_outputs: Any  # Omni-Specific
```

在 `execute_model()` 和 `sample_tokens()` 之间传递的临时状态。最后一个字段 `multimodal_outputs` 是 Omni 特有的扩展。

### 2. 初始化

```python
class NPUARModelRunner(OmniNPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, ...)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(...)
```

初始化时创建 input_ids 和 inputs_embeds 缓冲区，以及 KV 缓存传输管理器。

### 3. execute_model -- 模型前向传播

`execute_model()` 方法是核心推理入口，流程如下：

**Omni 特有逻辑**：
```python
# [Omni] 清除预热状态
if not getattr(self, "_warmup_state_cleared", False):
    self._warmup_state_cleared = True
    if hasattr(self.model, "_clear_warmup_state"):
        self.model._clear_warmup_state()

# [Omni] 处理 KV 缓存传输
self.kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(...)
```

**模型前向**：
```python
with set_ascend_forward_context(attn_metadata, self.vllm_config, ...):
    hidden_states = self._model_forward(num_tokens_padded, input_ids, positions, ...)

# [Omni] 提取多模态输出
hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
```

**Logits 计算**：
```python
sample_hidden_states = hidden_states[logits_indices]
try:
    logits = self.model.compute_logits(sample_hidden_states, sampling_metadata=...)
except TypeError:
    logits = self.model.compute_logits(sample_hidden_states)
```

方法返回 `None` 表示需要调用 `sample_tokens()` 完成采样。

### 4. sample_tokens -- 采样与输出

```python
@torch.inference_mode()
def sample_tokens(self, grammar_output):
    # 解包临时状态
    (scheduler_output, logits, ..., multimodal_outputs) = self.execute_model_state

    # [Omni] 修正 prompt_token_ids 越界问题
    if logits is not None and not self.input_batch.sampling_metadata.no_penalties:
        smd = self.input_batch.sampling_metadata
        if smd.prompt_token_ids is not None:
            logits_vocab = logits.shape[-1]
            if self.input_batch.vocab_size > logits_vocab:
                smd.prompt_token_ids = smd.prompt_token_ids.clamp(max=logits_vocab)

    # [Omni] 构建每请求的 pooler_output（包含 hidden_states 和多模态数据）
    for rid in req_ids_output_copy:
        idx = req_id_to_index_output_copy[rid]
        start = int(self.query_start_loc.cpu[idx])
        sched = int(num_scheduled_tokens_np[idx])
        end = start + sched
        hidden_slice = hidden_states_cpu[start:end]
        payload = {"hidden": hidden_slice}
        if isinstance(multimodal_outputs, dict) and multimodal_outputs:
            # 按请求切片多模态输出
            ...
        pooler_output.append(payload)

    model_runner_output = OmniModelRunnerOutput(
        req_ids=req_ids_output_copy,
        sampled_token_ids=valid_sampled_token_ids,
        pooler_output=pooler_output if engine_output_type != "text" else None,
        ...
    )
    model_runner_output.kv_extracted_req_ids = kv_extracted_req_ids
```

Omni AR 模型的输出不仅包含采样的 token IDs，还包含每个请求的 hidden states 和多模态数据，这些数据通过 `pooler_output` 传递给下游阶段。

### 5. 请求 ID 解析

```python
def _resolve_global_request_id(self, req_id: str) -> str:
    req_state = self.requests.get(req_id)
    if not req_state:
        return req_id
    add_info = self.model_intermediate_buffer.get(req_id, {})
    global_id = add_info.get("global_request_id")
    ...
```

将阶段内部请求 ID 映射到全局请求 ID，用于跨阶段的 KV 缓存传输。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ExecuteModelState` | NamedTuple | execute_model 和 sample_tokens 间的临时状态 |
| `NPUARModelRunner` | 类 | NPU 自回归 ModelRunner |
| `execute_model()` | 方法 | 模型前向传播，返回 None 或 IntermediateTensors |
| `sample_tokens()` | 方法 | 采样并构建 OmniModelRunnerOutput |
| `_resolve_global_request_id()` | 方法 | 解析全局请求 ID |

## 与其他模块的关系

- **基类**：`OmniNPUModelRunner`（npu_model_runner.py）
- **输出类型**：`OmniModelRunnerOutput`（vllm_omni.outputs）
- **KV 传输**：`OmniKVTransferManager`（vllm_omni.distributed）
- **使用者**：`NPUARWorker`（npu_ar_worker.py）

## 总结

`NPUARModelRunner` 是 NPU 平台自回归推理的核心实现，在标准 vLLM 推理流程基础上添加了大量 Omni 特有逻辑：多模态输出提取与按请求切片、KV 缓存跨阶段传输、hidden states 传递、prompt_token_ids 越界修正等。它是理解 vllm-omni 如何在 NPU 上实现多阶段多模态推理的关键文件。
