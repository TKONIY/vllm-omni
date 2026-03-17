# `omni_generation_scheduler.py` — 生成/扩散调度器

## 文件概述

`OmniGenerationScheduler` 是为非自回归生成模型（如扩散模型、TTS 模型）设计的调度器。其核心特点是"一步完成"——将所有输入 token 一次性发送给模型，模型在一步中完成推理并返回结果。

## 关键代码解析

### schedule() — 快速路径调度

```python
def schedule(self) -> SchedulerOutput:
    """扩散快速路径：
    - 一次性发送所有输入 token
    - 如果 token 数为 0，分配 1 个占位 token
    - 如果预算不足，回退到默认调度
    """
```

调度流程分为两个阶段：

**阶段 1：调度已运行的请求（running queue）**

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    required_tokens = len(request.prompt_token_ids) - num_computed_tokens
    # 异步分块模式：上游完成时追加占位 token
    if required_tokens <= 0:
        if self.chunk_transfer_adapter and request_id in self.chunk_transfer_adapter.finished_requests:
            request.prompt_token_ids.append(0)
            # ...
    num_new_tokens = min(required_tokens, token_budget)
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
```

**阶段 2：调度等待中的请求（waiting queue）**

```python
while self.waiting and token_budget > 0 and len(self.running) < self.max_num_running_reqs:
    request = self.waiting.peek_request()
    required_tokens = max(len(request.prompt_token_ids), 1)
    num_new_tokens = min(required_tokens, token_budget)
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    # 分配成功后移入 running
    request = self.waiting.pop_request()
    self.running.append(request)
```

**回退机制**：

```python
if not num_scheduled_tokens:
    if self.chunk_transfer_adapter:
        self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
    else:
        return super().schedule()  # 回退到 vLLM 默认调度
```

### update_from_output — 立即完成

```python
def update_from_output(self, scheduler_output, model_runner_output):
    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        # 扩散请求：一步完成
        if (
            request.status == RequestStatus.FINISHED_STOPPED
            or request.num_computed_tokens >= request.num_prompt_tokens
            or (self.chunk_transfer_adapter and request_id in finished_requests
                and request.num_computed_tokens >= len(request.prompt_token_ids))
        ):
            request.status = RequestStatus.FINISHED_STOPPED
            stopped = True
```

关键区别于 AR 调度器：
- 请求在预填充完成后**立即标记为完成**
- 不进行逐 token 解码
- 支持异步分块模式下的延迟完成判断

### OmniCachedRequestData

```python
cached_reqs_data = OmniCachedRequestData(
    prompt_token_ids=cached_prompt_token_ids,
    additional_information=cached_additional_information,
    # ... 基类字段
)
```

扩展缓存请求数据，携带 prompt token IDs 和附加信息，供模型执行器在连续推理步中使用。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniGenerationScheduler` | 类 | 生成/扩散模型调度器 |
| `schedule` | 方法 | 快速路径调度（一次性分配所有 token） |
| `update_from_output` | 方法 | 处理输出并立即完成请求 |

## 与其他模块的关系

- 继承 `vllm.v1.core.sched.scheduler.Scheduler`
- 使用 `output.py` 中的 `OmniNewRequestData`、`OmniCachedRequestData`
- 使用 `OmniChunkTransferAdapter` 处理异步分块
- 接收 `OmniModelRunnerOutput`（含多模态输出）

## 总结

`OmniGenerationScheduler` 通过"一步完成"的快速路径调度策略，优化了非自回归模型的推理效率。异步分块支持使其能处理流式上游输入（如 thinker 输出流式送入 talker），并在上游完成后自动触发最终推理。
