# `gpu_generation_model_runner.py` — 非自回归生成模型运行器

## 文件概述

`gpu_generation_model_runner.py` 定义了 `GPUGenerationModelRunner`，用于**非自回归生成阶段**（如 Qwen3-Omni 的 Code2Wav 音频合成）。与自回归的 `GPUARModelRunner` 不同，该运行器：

- **不计算 logits**
- **不进行 token 采样**
- 直接执行生成模型（如声码器），通过 `pooler_output` 返回生成结果（音频波形等）

## 关键代码解析

### 类定义

```python
class GPUGenerationModelRunner(OmniGPUModelRunner):
    """Generation model runner for vLLM-Omni (non-autoregressive).

    - Reuses GPUModelRunner preparation, multimodal handling, and TP/PP/DP glue.
    - Does not compute logits or perform token sampling.
    - Executes generation process and returns tensors via `pooler_output`.
    """
```

### _update_request_states — 请求状态更新

```python
def _update_request_states(self, scheduler_output: SchedulerOutput):
    # 移除已完成的请求
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)
    # 移除未调度的请求
    unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
    for req_id in unscheduled_req_ids:
        self.input_batch.remove_request(req_id)
    # 重新添加缓存的请求（带更新的 prompt_token_ids）
    for req_id in cached_reqs.req_ids:
        req_state.prompt_token_ids = cached_reqs.prompt_token_ids.get(req_id)
        self.input_batch.remove_request(req_id)
        self.input_batch.add_request(req_state)
```

该方法用于 `async_chunk` 模式下的请求状态刷新。非自回归生成模型的输入可能在每个步骤中变化（例如新的编码码流到达），所以需要移除并重新添加请求以更新其 `prompt_token_ids`。

### execute_model — 模型执行（第一阶段）

```python
@torch.inference_mode()
def execute_model(self, scheduler_output, intermediate_tensors=None):
```

流程与 `GPUARModelRunner` 类似，但有关键区别：

1. **async_chunk 支持**：
   ```python
   if self.model_config.async_chunk and num_scheduled_tokens:
       self._update_request_states(scheduler_output)
   ```

2. **传递 seq_token_counts**：
   ```python
   model_kwargs["seq_token_counts"] = tokens
   ```
   将每个请求的 token 数传给模型，用于 Code2Wav 的输出切片。

3. **调用 _run_generation_model 而非计算 logits**：
   ```python
   outputs = self._run_generation_model(
       input_ids=input_ids, positions=positions,
       intermediate_tensors=intermediate_tensors,
       inputs_embeds=inputs_embeds, model_kwargs=model_kwargs,
       logits_indices=logits_indices,
   )
   ```

4. **不计算 logits 和 sample_hidden_states**：
   ```python
   self.execute_model_state = ExecuteModelState(
       scheduler_output, None,  # logits=None
       ..., None, None, None,   # hidden_states, sample_hidden_states, aux=None
       ..., multimodal_outputs, slot_mappings,
   )
   ```

### sample_tokens — 输出打包（第二阶段）

```python
@torch.inference_mode()
def sample_tokens(self, grammar_output=None):
```

虽然叫 `sample_tokens`，但实际上不做采样，而是将生成结果打包为 `OmniModelRunnerOutput`：

```python
# 支持三种输出格式
if isinstance(multimodal_outputs, torch.Tensor):
    # 单个 Tensor：逐请求切片
    pooler_output.append({"model_outputs": multimodal_outputs[i].detach().to("cpu")})
elif isinstance(multimodal_outputs, list):
    # 列表：逐元素提取
    pooler_output.append({"model_outputs": out.detach().to("cpu")})
elif isinstance(multimodal_outputs, dict):
    # 字典：按键逐请求切片
    for key, out in multimodal_outputs.items():
        mm_payload[key] = out[i].detach().to("cpu")
    pooler_output.append(mm_payload)
```

最终输出：

```python
output = OmniModelRunnerOutput(
    req_ids=req_ids_output_copy,
    sampled_token_ids=[],     # 无采样结果
    logprobs=None,            # 无 logprobs
    pooler_output=pooler_output,  # 生成结果在这里
    ...
)
```

### _run_generation_model — 生成模型前向传播

```python
def _run_generation_model(self, *, input_ids, positions, intermediate_tensors,
                           inputs_embeds, model_kwargs, logits_indices):
    kwargs = dict(
        input_ids=input_ids, positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds, **model_kwargs,
        sampling_metadata=..., logits_index=..., sampler=...,
    )
    if hasattr(self.model, "forward"):
        return self._model_forward(**kwargs)
```

保持与 AR runner 相同的参数签名调用 `_model_forward`，确保 Omni 的 kwargs 注入正常工作。

### _dummy_run — 虚拟运行（Profile/CUDA Graph）

该方法与 `OmniGPUModelRunner._dummy_run()` 功能类似，但特别处理了：

```python
# 某些生成模型在 profiling 运行中需要运行时额外信息
if hasattr(self.model, "get_dummy_runtime_additional_information"):
    runtime_addi = self.model.get_dummy_runtime_additional_information(num_reqs)
    model_kwargs["runtime_additional_information"] = runtime_addi
```

例如 MammothModa2 扩散模型需要图像尺寸和条件 embedding 等信息才能正确执行 profiling。

### profile_run — 显存剖析

```python
def profile_run(self) -> None:
    if self.supports_mm_inputs:
        # 运行多模态编码器 profiling
        ...
    hidden_states, _ = self._dummy_run(self.max_num_tokens, is_profile=True)
    del hidden_states
    self.encoder_cache.clear()
    gc.collect()
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `GPUGenerationModelRunner` | 类 | 非自回归生成模型运行器 |
| `execute_model()` | 方法 | 执行生成模型前向传播 |
| `sample_tokens()` | 方法 | 将多模态生成结果打包为输出（不做采样） |
| `_run_generation_model()` | 方法 | 调用模型 forward 生成结果 |
| `_update_request_states()` | 方法 | async_chunk 模式下刷新请求状态 |
| `_dummy_run()` | 方法 | Profile/CUDA Graph 虚拟运行 |
| `profile_run()` | 方法 | 显存使用剖析 |
| `_dummy_sampler_run()` | 方法 | 空操作（生成模型无需采样器） |

## 与其他模块的关系

- **继承** `OmniGPUModelRunner`（`gpu_model_runner.py`）
- **被使用** `GPUGenerationWorker`（`gpu_generation_worker.py`）创建并管理此 Runner
- **复用** `ExecuteModelState`（来自 `gpu_ar_model_runner.py`）
- **输出** `OmniModelRunnerOutput`（`vllm_omni.outputs`）

## 总结

`GPUGenerationModelRunner` 填补了 vLLM-Omni 多阶段流水线中非自回归生成阶段的需求。它复用了上游 `GPUModelRunner` 的所有基础设施（批次管理、注意力元数据、CUDA Graph 等），但跳过了 logits 计算和 token 采样，直接将模型生成的多模态结果（如音频波形）通过 `pooler_output` 返回。它支持三种输出格式（Tensor、列表、字典），能够灵活适配不同类型的生成模型。
