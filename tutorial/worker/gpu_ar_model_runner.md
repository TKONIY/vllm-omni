# `gpu_ar_model_runner.py` — 自回归 GPU 模型运行器

## 文件概述

`gpu_ar_model_runner.py` 定义了 `GPUARModelRunner`，这是 vLLM-Omni 中用于**自回归（Autoregressive）文本生成**阶段的模型运行器。它继承自 `OmniGPUModelRunner`，核心职责是执行模型前向传播、采样 token，并将每个请求的 hidden states 和多模态输出暴露给下游阶段。

## 关键代码解析

### ExecuteModelState — 两阶段状态载体

```python
class ExecuteModelState(NamedTuple):
    scheduler_output: SchedulerOutput
    logits: torch.Tensor | None
    spec_decode_metadata: Any
    spec_decode_common_attn_metadata: Any
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: Any
    cudagraph_stats: Any
    multimodal_outputs: Any          # Omni 扩展字段
    slot_mappings: dict | list | None  # Omni 扩展字段
```

该 NamedTuple 用于在 `execute_model()` 和 `sample_tokens()` 之间传递中间状态。上游 vLLM 采用两阶段（execute + sample）架构，Omni 在此基础上添加了 `multimodal_outputs` 和 `slot_mappings` 字段。

### 初始化

```python
class GPUARModelRunner(OmniGPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False
        )
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(
            self.vllm_config, self.model_config
        )
```

关键初始化：
- 预分配 `input_ids` 和 `inputs_embeds` 缓冲区（用于 CUDA Graph 兼容）
- 初始化 `OmniKVTransferManager` 用于跨阶段 KV Cache 传输

### _make_buffer — Ray 兼容的缓冲区分配

```python
def _make_buffer(self, *size, dtype, numpy=True):
    from vllm_omni.distributed.ray_utils.utils import (
        calculate_total_bytes, maybe_disable_pin_memory_for_ray,
    )
    total_bytes = calculate_total_bytes(size, dtype)
    with maybe_disable_pin_memory_for_ray(self, total_bytes):
        return super()._make_buffer(*size, dtype=dtype, numpy=numpy)
```

防止 Ray 对大缓冲区进行不必要的内存钉定（pin），避免性能问题。

### execute_model — 模型执行（第一阶段）

```python
@torch.inference_mode()
def execute_model(self, scheduler_output, intermediate_tensors=None):
```

主要流程：

1. **KV Transfer 处理**：在更新状态之前，处理已完成请求的 KV Cache 传输
   ```python
   self.kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(...)
   ```

2. **状态更新与输入准备**：调用 `_update_states()` 和 `_prepare_inputs()` 准备批次数据

3. **注意力元数据构建**：构建 slot_mappings 和 attention metadata

4. **预处理**：调用 `_preprocess()` 处理多模态输入、prompt embeds 等

5. **模型前向传播**：
   ```python
   model_output = self._model_forward(
       input_ids=input_ids, positions=positions,
       intermediate_tensors=intermediate_tensors,
       inputs_embeds=inputs_embeds, **model_kwargs,
       sampling_metadata=..., logits_index=..., sampler=...,
   )
   ```

6. **后处理**：提取多模态输出、计算 logits
   ```python
   hidden_states, multimodal_outputs = self.extract_multimodal_outputs(model_output)
   sample_hidden_states = hidden_states[logits_indices]
   logits = self.model.compute_logits(sample_hidden_states, ...)
   ```

7. **保存状态**：将所有中间结果打包到 `ExecuteModelState` 中，返回 `None` 表示需要调用 `sample_tokens()`

### sample_tokens — Token 采样（第二阶段）

```python
@torch.inference_mode()
def sample_tokens(self, grammar_output=None):
```

主要流程：

1. **结构化输出约束**：如果有 grammar_output，应用 bitmask 约束

2. **采样**：
   ```python
   sampler_output = self._sample(logits, spec_decode_metadata)
   ```

3. **投机解码**：如果启用了 speculative decoding，运行 drafter 生成草稿 token

4. **簿记同步**：调用 `_bookkeeping_sync()` 完成 logprobs 计算、token 验证等

5. **Hidden States 提取**（Omni 核心特性）：
   ```python
   hidden_states_cpu = hidden_states.detach().to("cpu").contiguous()
   # 为每个请求切片 hidden states
   for rid in req_ids_output_copy:
       idx = req_id_to_index_output_copy[rid]
       start = int(self.query_start_loc.cpu[idx])
       sched = int(num_scheduled_tokens_np[idx])
       hidden_slice = hidden_states_cpu[start:start+sched]
       payload = {"hidden": hidden_slice}
       # 合并多模态输出
       ...
       pooler_output.append(payload)
   ```

6. **构建输出**：返回 `OmniModelRunnerOutput`，其中 `pooler_output` 包含每个请求的 hidden states 和多模态输出

### _resolve_global_request_id — 全局请求 ID 解析

```python
def _resolve_global_request_id(self, req_id: str) -> str:
    req_state = self.requests.get(req_id)
    add_info = self.model_intermediate_buffer.get(req_id, {})
    global_id = add_info.get("global_request_id")
    ...
```

在多阶段流水线中，每个阶段可能使用不同的本地请求 ID。该方法从 `model_intermediate_buffer` 中查找全局请求 ID，用于跨阶段的 KV Cache 传输。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `ExecuteModelState` | NamedTuple | execute_model 和 sample_tokens 之间的状态载体 |
| `GPUARModelRunner` | 类 | 自回归模型运行器 |
| `execute_model()` | 方法 | 第一阶段：执行模型前向传播，计算 logits |
| `sample_tokens()` | 方法 | 第二阶段：采样 token，提取 hidden states |
| `_make_buffer()` | 方法 | Ray 兼容的缓冲区分配 |
| `_resolve_global_request_id()` | 方法 | 解析跨阶段的全局请求 ID |

## 与其他模块的关系

- **继承** `OmniGPUModelRunner`（`gpu_model_runner.py`）
- **被使用** `GPUARWorker`（`gpu_ar_worker.py`）创建并管理此 Runner
- **输出** `OmniModelRunnerOutput`（`vllm_omni.outputs`），包含采样结果和 hidden states
- **依赖** `OmniKVTransferManager`（`vllm_omni.distributed`）处理跨阶段 KV Cache 传输
- **依赖** `ExecuteModelState` 被 `GPUGenerationModelRunner` 复用

## 总结

`GPUARModelRunner` 是 vLLM-Omni 中自回归推理阶段的核心组件。相比上游 `GPUModelRunner`，它的关键差异在于：

1. **暴露 hidden states**：通过 `pooler_output` 将每个请求的 hidden states 传递给下游阶段
2. **多模态输出合并**：将模型产生的多模态输出（音频编码等）与 hidden states 一起打包
3. **KV Transfer 管理**：支持跨阶段的 KV Cache 传输，减少重复计算
4. **Ray 兼容性**：防止 Ray 对大缓冲区的内存钉定问题

这使得它能够作为多阶段推理流水线中的"思考"（thinker）阶段运行。
