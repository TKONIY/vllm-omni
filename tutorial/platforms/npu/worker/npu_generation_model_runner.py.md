# `npu_generation_model_runner.py` -- NPU 生成 ModelRunner

## 文件概述

`npu_generation_model_runner.py` 实现了 `NPUGenerationModelRunner`，用于 Omni 模型的非自回归生成阶段（如 code2wav，将 codec codes 转换为音频波形）。与 AR ModelRunner 不同，该类不进行 token 采样，而是直接运行生成模型并收集多模态输出。

## 关键代码解析

### 1. 请求状态更新

```python
def _update_request_states(self, scheduler_output: SchedulerOutput):
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)
    # 处理 async_chunk 模式下的请求更新
    cached_reqs = scheduler_output.scheduled_cached_reqs
    for req_id in cached_reqs.req_ids:
        req_state = self.requests.get(req_id)
        req_state.prompt_token_ids = cached_reqs.prompt_token_ids.get(req_id)
        if req_id in self.input_batch.req_id_to_index:
            self.input_batch.remove_request(req_id)
    for req_state in req_states:
        self.input_batch.add_request(req_state)
```

该方法专门处理 `async_chunk` 模式下的请求状态管理，在流式生成中需要不断更新输入批次。

### 2. execute_model -- 生成模型前向传播

核心差异在于使用 `_run_generation_model` 替代标准的 `_model_forward`：

```python
with set_ascend_forward_context(...):
    outputs = self._run_generation_model(
        num_tokens_padded=num_tokens_padded,
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        model_kwargs=model_kwargs,
        logits_indices=logits_indices,
    )
    _, multimodal_outputs = self.extract_multimodal_outputs(outputs)
```

注意 `ExecuteModelState` 中 `logits`、`hidden_states`、`sample_hidden_states`、`aux_hidden_states` 均为 `None`，因为生成模型不需要这些中间结果。

### 3. sample_tokens -- 收集生成输出

```python
def sample_tokens(self, grammar_output):
    # 不执行采样，直接处理多模态输出
    if isinstance(multimodal_outputs, torch.Tensor):
        for i in range(self.input_batch.num_reqs):
            pooler_output.append({"model_outputs": multimodal_outputs[i].detach().to("cpu").contiguous()})
    elif isinstance(multimodal_outputs, list):
        for out in multimodal_outputs:
            pooler_output.append({"model_outputs": out.detach().to("cpu").contiguous() if out is not None else None})
    elif isinstance(multimodal_outputs, dict):
        for i in range(num_reqs):
            mm_payload = {}
            for key, out in multimodal_outputs.items():
                if isinstance(out, list):
                    mm_payload[key] = out[i].detach().to("cpu").contiguous()
                elif isinstance(out, torch.Tensor):
                    mm_payload[key] = out.detach().to("cpu").contiguous()
            pooler_output.append(mm_payload)

    output = OmniModelRunnerOutput(
        sampled_token_ids=[],  # 生成模型无采样 token
        pooler_output=pooler_output,
        ...
    )
```

生成模型的输出完全通过 `pooler_output` 传递，`sampled_token_ids` 为空列表。支持三种输出格式：
- `torch.Tensor`：单一张量，按请求索引切片
- `list`：每个元素对应一个请求的输出
- `dict`：多键输出，每个键按请求切片

### 4. _run_generation_model

```python
def _run_generation_model(self, *, num_tokens_padded, input_ids, positions, ...):
    kwargs = dict(
        num_tokens_padded=num_tokens_padded,
        input_ids=input_ids, positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs,
        sampling_metadata=self.input_batch.sampling_metadata,
        logits_index=logits_indices,
        sampler=self.sampler,
    )
    if hasattr(self.model, "forward"):
        return self._model_forward(**kwargs)
```

将 `sampling_metadata`、`logits_index` 和 `sampler` 作为额外参数传递给模型，支持生成模型内部的采样需求。

### 5. _dummy_run 与 profile_run

Generation ModelRunner 的 `_dummy_run` 方法增加了对 `get_dummy_runtime_additional_information` 的支持：

```python
if hasattr(self.model, "get_dummy_runtime_additional_information"):
    runtime_addi = self.model.get_dummy_runtime_additional_information(num_reqs)
    model_kwargs["runtime_additional_information"] = runtime_addi
```

这允许生成模型（如图像扩散模型）在 profiling 阶段提供占位输入。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `NPUGenerationModelRunner` | 类 | NPU 非自回归生成 ModelRunner |
| `_update_request_states()` | 方法 | async_chunk 模式请求状态更新 |
| `execute_model()` | 方法 | 生成模型前向传播 |
| `sample_tokens()` | 方法 | 收集多模态输出（无采样） |
| `_run_generation_model()` | 方法 | 调用生成模型前向 |
| `_dummy_run()` | 方法 | profiling 和图捕获 |
| `profile_run()` | 方法 | 内存估算入口 |

## 与其他模块的关系

- **基类**：`OmniNPUModelRunner`（npu_model_runner.py）
- **状态复用**：使用 `npu_ar_model_runner.py` 中定义的 `ExecuteModelState`
- **输出类型**：`OmniModelRunnerOutput`
- **使用者**：`NPUGenerationWorker`

## 总结

`NPUGenerationModelRunner` 针对非自回归生成场景（如音频合成、图像生成）优化了推理流程。它不执行 token 采样，而是直接将生成模型的多模态输出按请求切片后打包为 `pooler_output`。通过 `async_chunk` 请求状态管理和 dummy runtime 信息支持，它能够处理流式音频生成和扩散模型推理等复杂场景。
