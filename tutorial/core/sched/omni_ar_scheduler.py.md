# `omni_ar_scheduler.py` — AR 自回归调度器

## 文件概述

`OmniARScheduler` 扩展 vLLM 的调度器，为自回归（AR）模型阶段添加 KV 缓存跨阶段传输和异步分块处理支持。这是 vllm-omni 中最复杂的调度器组件。

## 关键代码解析

### 初始化 — KV 传输状态管理

```python
class OmniARScheduler(VLLMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests_needing_kv_transfer: dict[str, dict[str, Any]] = {}
        self.waiting_for_transfer_free: set[str] = set()
        self.active_kv_transfers: set[str] = set()
        self.kv_transfer_criteria = self._get_kv_transfer_criteria()
        self.transfer_triggered_requests: set[str] = set()
```

维护四个状态集合来管理 KV 缓存传输的生命周期：
- `requests_needing_kv_transfer`：待传输队列
- `active_kv_transfers`：正在传输中
- `waiting_for_transfer_free`：传输完成，等待释放内存
- `transfer_triggered_requests`：已触发传输（防重复）

### KV 传输触发机制

```python
def _process_kv_transfer_trigger(self, request, new_token_ids):
    if criteria_type == "prefill_finished":
        if request.num_computed_tokens >= request.num_prompt_tokens:
            self._mark_request_for_kv_transfer(request.request_id, request.num_computed_tokens)
            return False  # 不停止请求，继续解码

    elif criteria_type == "special_token":
        target_token_id = self.kv_transfer_criteria.get("token_id")
        if target_token_id in new_token_ids:
            # 精确计算快照长度（截至特殊 token）
            idx = new_token_ids.index(target_token_id)
            snapshot_len = request.num_computed_tokens - (len(new_token_ids) - (idx + 1))
            self._mark_request_for_kv_transfer(request.request_id, snapshot_len)
            return False
```

支持两种传输触发条件：
1. **`prefill_finished`**：预填充完成时触发，请求继续解码
2. **`special_token`**：检测到特殊 token 时触发，精确截断 KV 缓存

### schedule() 方法增强

```python
def schedule(self):
    # 1. 处理异步分块
    if self.chunk_transfer_adapter:
        self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)

    # 2. 调用基类调度
    scheduler_output = super().schedule()

    # 3. 将 NewRequestData 包装为 OmniNewRequestData
    for nr in scheduler_output.scheduled_new_reqs:
        omni_nr = OmniNewRequestData(
            prompt_embeds=getattr(request, "prompt_embeds", None),
            additional_information=getattr(request, "additional_information", None),
            # ... 其他字段
        )

    # 4. 包装为 OmniSchedulerOutput（携带 KV 传输元数据）
    return OmniSchedulerOutput(
        **base_data,
        finished_requests_needing_kv_transfer=finished_reqs,
    )
```

### _free_request — 延迟释放

```python
def _free_request(self, request, delay_free_blocks=False):
    if self._should_transfer_kv_for_request(request_id):
        if already_triggered and is_active:
            # 传输还在进行，延迟释放
            self.waiting_for_transfer_free.add(request_id)
            return kv_xfer_params
        elif not already_triggered:
            # 首次标记，发起传输
            self.waiting_for_transfer_free.add(request_id)
            self._mark_request_for_kv_transfer(request_id, request.num_computed_tokens)
            return kv_xfer_params
    # 标准释放
    super()._free_blocks(request)
```

关键设计：当请求完成但 KV 缓存仍需传输时，延迟释放内存块，直到传输确认完成。

### update_from_output — 处理模型输出

在基类逻辑之上增加：
- KV 传输触发检查
- 异步分块保存
- `WAITING_FOR_CHUNK` 状态处理
- KV 提取确认后的内存释放

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniARScheduler` | 类 | AR 模型的调度器 |
| `KVCacheTransferData` | 数据类 | KV 缓存传输数据包 |
| `schedule` | 方法 | 调度请求，包装 omni 元数据 |
| `update_from_output` | 方法 | 处理模型输出，管理 KV 传输 |
| `_process_kv_transfer_trigger` | 方法 | 检查并触发 KV 传输 |
| `_mark_request_for_kv_transfer` | 方法 | 标记请求需要 KV 传输 |
| `_free_request` | 方法 | 释放请求资源（支持延迟释放） |
| `has_unfinished_requests` | 方法 | 检查是否有未完成请求（含传输中） |

## 与其他模块的关系

- 继承 `vllm.v1.core.sched.scheduler.Scheduler`
- 使用 `output.py` 中的 `OmniNewRequestData` 和 `OmniSchedulerOutput`
- 使用 `distributed.omni_connectors` 中的 `OmniChunkTransferAdapter`
- 通过 `vllm_config.model_config.omni_kv_config` 读取 KV 传输配置

## 总结

`OmniARScheduler` 是 vllm-omni 调度系统的核心，在 vLLM 标准调度流程上叠加了 KV 缓存跨阶段传输和异步分块处理能力。通过精细的状态管理确保传输完成前不释放内存，同时保持与 vLLM 调度逻辑的兼容性。
