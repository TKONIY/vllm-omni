# `adapter.py` — Orchestrator 级别的连接器适配函数

## 文件概述

该文件提供 Orchestrator（编排器）层面的发送和接收适配函数，封装了通过 OmniConnector 在阶段之间传输数据的逻辑。它还包含一个用于计算 Talker 模型 prompt token 长度的工具函数。

## 关键代码解析

### 1. try_send_via_connector — 发送数据

```python
def try_send_via_connector(
    connector, stage_id, next_stage_id, req_id,
    next_inputs, sampling_params, original_prompt,
    next_stage_queue_submit_fn, metrics,
) -> bool:
```

发送流程：
1. **清理不可序列化字段**：从 `original_prompt` 中移除 `mm_kwargs`、`mm_placeholders`、`mm_hashes` 等多模态特征字段，因为它们包含 `MultiModalKwargsItems` 对象，不被 `OmniMsgpackEncoder` 支持
2. **构建 payload**：包含 `engine_inputs`、`sampling_params` 和 metadata
3. **调用 `connector.put()`**：发送数据到连接器
4. **发送轻量级通知**：成功后通过 `next_stage_queue_submit_fn` 向下游队列提交一个通知消息（包含 `from_connector=True` 标记和可能的 `connector_metadata`）
5. **记录指标**：通过 `metrics.on_forward()` 记录传输时间和字节数

```python
notify_payload = {
    "type": OmniStageTaskType.GENERATE,
    "request_id": req_id,
    "from_connector": True,
    "from_stage": str(stage_id),
    "to_stage": str(next_stage_id),
    "connector_metadata": metadata,  # 如有
}
next_stage_queue_submit_fn(notify_payload)
```

### 2. try_recv_via_connector — 接收数据

```python
def try_recv_via_connector(task, connectors, stage_id) -> tuple[Any, dict | None]:
```

接收流程：
1. 检查 `task["from_connector"]` 标记
2. 根据 `from_stage` 和当前 `stage_id` 查找对应的连接器
3. 调用 `connector.get()` 从连接器获取数据，传入 `connector_metadata`
4. 提取 `engine_inputs` 并返回接收指标

如果数据不是通过连接器传输的，则回退到 IPC 机制（`maybe_load_from_ipc_with_metrics`）。

### 3. compute_talker_prompt_ids_length — Talker prompt 长度计算

```python
def compute_talker_prompt_ids_length(prompt_ids: list[int]) -> int:
```

解析 prompt token ID 序列，根据特殊 token（`im_start`、`system`、`user`、`assistant`）识别对话轮次，累计 user 部分长度和 assistant 固定长度（9 个 token）。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `try_send_via_connector` | function | 通过连接器发送数据到下一阶段 |
| `try_recv_via_connector` | function | 从连接器接收数据 |
| `compute_talker_prompt_ids_length` | function | 计算 Talker 模型的 prompt 长度 |

## 与其他模块的关系

- 使用 `OmniConnectorBase.put()` / `get()` 接口
- 与 `entrypoints/stage_utils.py` 中的 IPC 工具函数配合
- 被 Orchestrator（`omni_orchestrator.py`）在阶段转发时调用
- 使用 `OrchestratorAggregator` 记录指标

## 总结

`adapter.py` 是 Orchestrator 与 OmniConnector 之间的桥梁，封装了发送/接收的完整流程（包括序列化安全处理、轻量级通知、指标记录、异常回退）。
