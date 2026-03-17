# `chunk_transfer_adapter.py` — 分块传输适配器

## 文件概述

该文件实现了 `OmniChunkTransferAdapter`，是 `OmniTransferAdapterBase` 的核心子类。它管理阶段间按分块（chunk）粒度的异步数据传输，支持 AR（自回归）和非 AR（如扩散模型）两种模式，并与 vLLM v1 调度器深度集成。

## 关键代码解析

### 1. 初始化

```python
class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    def __init__(self, vllm_config):
        self.connector = self.create_connector(model_config)
        super().__init__(model_config)  # 启动后台线程

        self.model_mode = getattr(model_config, "worker_type", None) or "ar"
        self.put_req_chunk: dict[str, int] = defaultdict(int)   # 发送侧 chunk 计数
        self.get_req_chunk: dict[str, int] = defaultdict(int)   # 接收侧 chunk 计数
        self.finished_requests: set[str] = set()
        self.request_payload = {}                                # 累计 payload
        self.code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)

        # 调度器集成
        self.waiting_for_chunk_waiting_requests: deque = deque()
        self.waiting_for_chunk_running_requests: deque = deque()
        self.requests_with_ready_chunks = set()
```

### 2. 连接器创建（类方法）

```python
@classmethod
def create_connector(cls, model_config):
    connector_config = getattr(model_config, "stage_connector_config", None)
    connector_specs = ConnectorSpec(
        name=connector_config.get("name", "SharedMemoryConnector"),
        extra=connector_config.get("extra", {}),
    )
    return OmniConnectorFactory.create_connector(connector_specs)
```

默认使用 `SharedMemoryConnector`。

### 3. 分块接收（_poll_single_request）

```python
def _poll_single_request(self, request):
    chunk_id = self.get_req_chunk[req_id]
    connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"
    result = self.connector.get(str(target_stage_id), str(stage_id), connector_get_key)

    if result:
        self.get_req_chunk[req_id] += 1  # chunk 计数器递增

        if self.model_mode == "ar":
            # AR 模式：累计 payload（连接 tensor、合并 list）
            self._update_request_payload(external_req_id, payload_data)
            request.additional_information = payload_data
        else:
            # 非 AR 模式：替换 prompt_token_ids
            request.prompt_token_ids = payload_data.get("code_predictor_codes", [])
            request.num_computed_tokens = 0

        if payload_data.get("finished"):
            self.finished_requests.add(req_id)
```

### 4. Payload 累计逻辑（AR 模式）

```python
def _update_request_payload(self, req_id, payload_data):
    origin_payload = self.request_payload[req_id]
    for key, value in payload_data.items():
        if key in override_keys:
            payload_data[key] = value                              # 覆盖
        elif isinstance(value, torch.Tensor) and key in origin_payload:
            payload_data[key] = torch.cat([origin_payload[key], value], dim=0)  # 连接 tensor
        elif isinstance(value, list) and key in origin_payload:
            payload_data[key] = origin_payload[key] + value        # 合并 list
```

### 5. 分块发送（_send_single_request）

```python
def _send_single_request(self, task):
    connector_put_key = f"{request_id}_{stage_id}_{chunk_id}"
    # 调用自定义处理函数生成 payload
    payload_data = self.custom_process_next_stage_input_func(
        transfer_manager=self, pooling_output=pooling_output,
        request=request, is_finished=is_finished,
    )
    self.connector.put(str(stage_id), str(next_stage_id), connector_put_key, payload_data)
    self.put_req_chunk[request_id] += 1
```

### 6. 调度器集成

```python
def process_pending_chunks(self, waiting_queue, running_queue):
    """在调度前处理分块状态转换"""
    # 对 waiting 和 running 队列中的请求：
    # - 未加载的请求 → WAITING_FOR_CHUNK，移到等待列表
    # - 已完成加载的请求 → 恢复为 WAITING/RUNNING
    self._process_chunk_queue(waiting_queue, ..., RequestStatus.WAITING, ...)
    self._process_chunk_queue(running_queue, ..., RequestStatus.RUNNING, ...)
    # 限制 running 队列不超过 max_num_seqs
```

### 7. 资源清理

```python
def cleanup(self, request_id, external_req_id=None):
    """清理所有 per-request 状态"""
    self.finished_requests.discard(request_id)
    self.get_req_chunk.pop(request_id, None)
    self.requests_with_ready_chunks.discard(request_id)
    # ... 清理所有相关字典和队列
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniChunkTransferAdapter` | class | 分块级别的传输适配器 |
| `create_connector()` | classmethod | 从模型配置创建连接器 |
| `load_async()` | method | 注册请求等待分块加载 |
| `save_async()` | method | 提交分块数据待发送 |
| `_poll_single_request()` | method | 轮询获取一个分块 |
| `_send_single_request()` | method | 发送一个分块 |
| `_update_request_payload()` | method | AR 模式下累计 payload |
| `process_pending_chunks()` | method | 调度器前的分块状态处理 |
| `restore_queues()` | method | 恢复等待分块的请求到调度队列 |
| `postprocess_scheduler_output()` | method | 调度后清理就绪标记 |
| `cleanup()` | method | 清理 per-request 状态 |

## 与其他模块的关系

- 继承 `OmniTransferAdapterBase`
- 使用 `OmniConnectorFactory` 创建连接器
- 与 vLLM v1 的 `Request`、`RequestStatus`、`Scheduler` 深度集成
- 支持自定义 `custom_process_next_stage_input_func`（通过 model_config 指定的模块路径动态加载）

## 总结

`OmniChunkTransferAdapter` 是连接器框架与 vLLM 调度器之间的桥梁。它实现了分块级别的异步传输——每个请求的数据可以分多次传输（多个 chunk），适配器自动累计数据直到收到 `finished` 标记。同时，它通过 `WAITING_FOR_CHUNK` 状态与调度器协作，确保只有数据就绪的请求才会被调度执行。
