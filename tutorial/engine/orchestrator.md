# `orchestrator.py` — 多阶段调度编排器

## 文件概述

`orchestrator.py` 实现了 `Orchestrator` 类，它是 vLLM-Omni 多阶段推理流水线的核心调度器。Orchestrator 运行在一个独立的后台线程中，拥有自己的 asyncio 事件循环，负责：接收来自 `AsyncOmniEngine` 的请求、将请求分发到各阶段的 `StageEngineCoreClient`、轮询阶段输出、在阶段之间转发数据、以及将最终结果返回给调用者。

## 关键代码解析

### 1. 从 token 构建请求

```python
def build_engine_core_request_from_tokens(
    request_id: str,
    prompt: dict[str, Any],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    model_config: ModelConfig | None = None,
) -> OmniEngineCoreRequest:
    prompt_token_ids = prompt["prompt_token_ids"]
    sampling_params = params.clone()
    if sampling_params.max_tokens is None and model_config is not None:
        sampling_params.max_tokens = model_config.max_model_len - len(prompt_token_ids)

    prompt_embeds = prompt.get("prompt_embeds")
    additional_info_payload = serialize_additional_information(
        prompt.get("additional_information"),
    )
    return OmniEngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        prompt_embeds=prompt_embeds,
        additional_information=additional_info_payload,
        # ...
    )
```

这是阶段间转发时的请求构建函数。与 Stage-0 使用完整的 `InputProcessor` 不同，后续阶段直接从上游输出的 token IDs 构建请求，跳过了 tokenization、多模态预处理等步骤。如果 `max_tokens` 未指定，自动计算为 `max_model_len - prompt_len`。

### 2. OrchestratorRequestState — 请求状态跟踪

```python
@dataclass
class OrchestratorRequestState:
    request_id: str
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
```

每个活跃请求对应一个 `OrchestratorRequestState`，记录：
- `prompt`：原始 prompt（用于下游阶段访问原始数据）
- `sampling_params_list`：每个阶段的采样参数
- `final_stage_id`：最终阶段编号（决定何时完成请求）
- `stage_submit_ts`：每个阶段的提交时间戳（用于性能指标）

### 3. Orchestrator 主循环

```python
class Orchestrator:
    async def run(self):
        request_task = asyncio.create_task(self._request_handler())
        output_task = asyncio.create_task(self._orchestration_output_handler())
        try:
            await asyncio.gather(request_task, output_task)
        finally:
            self._shutdown_stages()
```

Orchestrator 的事件循环包含两个并行运行的协程：
- **`_request_handler`**：从请求队列读取消息（add_request / abort / collective_rpc / shutdown）
- **`_orchestration_output_handler`**：轮询所有阶段的输出并进行路由

两者通过 `asyncio.gather()` 并发运行，任一任务异常都会触发全部关闭。

### 4. 输出轮询与路由循环

```python
async def _orchestration_loop(self):
    while not self._shutdown_event.is_set():
        idle = True
        for stage_id in range(self.num_stages):
            stage_client = self.stage_clients[stage_id]

            # Diffusion 阶段：非阻塞轮询
            if stage_client.stage_type == "diffusion":
                output = stage_client.get_diffusion_output_async()
                if output is not None:
                    await self._route_output(stage_id, output, req_state, ...)
                continue

            # LLM 阶段：带超时的异步轮询
            try:
                raw_outputs = await asyncio.wait_for(self._poll_stage_raw(stage_id), timeout=0.001)
            except asyncio.TimeoutError:
                continue

            # 处理原始输出
            request_outputs = await self._process_stage_outputs(stage_id, raw_outputs)

            # 路由每个处理后的输出
            for output in request_outputs:
                await self._route_output(stage_id, output, req_state, ...)

        if idle:
            await asyncio.sleep(0.001)  # 防止忙等
```

核心轮询逻辑依次检查每个阶段：
- Diffusion 阶段使用非阻塞队列获取输出
- LLM 阶段通过 `asyncio.wait_for` 添加 1ms 超时，避免阻塞其他阶段
- 空闲时 sleep 1ms 让出 CPU

### 5. 输出路由逻辑

```python
async def _route_output(self, stage_id, output, req_state, stage_metrics):
    stage_client = self.stage_clients[stage_id]

    # 如果当前阶段产出最终输出，发送到输出队列
    if stage_client.final_output:
        await self.output_async_queue.put({
            "type": "output",
            "request_id": req_id,
            "engine_outputs": output,
            "finished": finished and stage_id == req_state.final_stage_id,
        })

    # 如果请求完成且有下游阶段，转发到下一阶段
    if finished and stage_id < req_state.final_stage_id and not self.async_chunk:
        await self._forward_to_next_stage(req_id, stage_id, output, req_state)

    # 如果到达最终阶段且完成，清理请求状态
    if finished and stage_id == req_state.final_stage_id:
        self.request_states.pop(req_id, None)
```

路由策略：
- **产出输出**：标记了 `final_output` 的阶段会将结果发送到用户
- **阶段转发**：非 async-chunk 模式下，阶段完成时触发下游阶段
- **状态清理**：最终阶段完成时移除请求状态

### 6. 阶段间转发

```python
async def _forward_to_next_stage(self, req_id, stage_id, output, req_state):
    next_stage_id = stage_id + 1
    next_client = self.stage_clients[next_stage_id]

    # Diffusion 下游
    if next_client.stage_type == "diffusion":
        self.stage_clients[stage_id].set_engine_outputs([output])
        diffusion_prompt = next_client.custom_process_input_func(...)
        await next_client.add_request_async(req_id, diffusion_prompt, params)
        return

    # LLM 下游
    self.stage_clients[stage_id].set_engine_outputs([output])
    next_inputs = next_client.process_engine_inputs(
        stage_list=self.stage_clients,
        prompt=req_state.prompt,
    )
    for next_input in next_inputs:
        request = build_engine_core_request_from_tokens(
            request_id=req_id,
            prompt=next_input,
            params=params,
            model_config=self.stage_vllm_configs[next_stage_id].model_config,
        )
        await next_client.add_request_async(request)
```

转发过程：
1. 将当前阶段的输出设置到 stage_client 上
2. 下游 stage_client 的 `process_engine_inputs()` 读取上游输出并生成新的输入
3. 从 token 构建轻量请求并提交给下游

### 7. Async-Chunk 预热

```python
async def _prewarm_async_chunk_stages(self, request_id, stage0_request, req_state):
    next_prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
    base_input = {"prompt_token_ids": [0] * next_prompt_len}

    for next_stage_id in range(1, req_state.final_stage_id + 1):
        request = build_engine_core_request_from_tokens(
            request_id=request_id,
            prompt=base_input,
            params=params,
        )
        await next_client.add_request_async(request)
```

在 async-chunk 模式下，请求到达 Stage-0 时就立即"预热"下游阶段。使用占位符 token IDs（全零）构建请求提交到下游，实际数据通过共享内存连接器流式传输。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `build_engine_core_request_from_tokens()` | 函数 | 从 token IDs 构建轻量级请求（阶段间转发） |
| `OrchestratorRequestState` | 数据类 | 每请求状态跟踪（采样参数、时间戳、最终阶段） |
| `Orchestrator` | 类 | 核心调度器，管理阶段间请求转发与输出路由 |
| `Orchestrator.run()` | 方法 | 事件循环入口，启动请求处理和输出轮询 |
| `Orchestrator._orchestration_loop()` | 方法 | 输出轮询与路由的内部循环 |
| `Orchestrator._route_output()` | 方法 | 决定输出走向（返回用户/转发下游/清理状态） |
| `Orchestrator._forward_to_next_stage()` | 方法 | 阶段间数据转发逻辑 |
| `Orchestrator._prewarm_async_chunk_stages()` | 方法 | async-chunk 模式下预热下游阶段 |
| `Orchestrator._handle_collective_rpc()` | 方法 | 处理控制面 RPC 请求 |
| `Orchestrator._build_stage_metrics()` | 方法 | 构建阶段性能指标 |

## 与其他模块的关系

- **`async_omni_engine.py`**：创建 Orchestrator 实例并提供请求/输出队列
- **`stage_engine_core_client.py`**：Orchestrator 持有 `StageEngineCoreClient` 列表，向其提交请求和轮询输出
- **`output_processor.py`**：Orchestrator 调用 `MultimodalOutputProcessor.process_outputs()` 处理原始输出
- **`serialization.py`**：`build_engine_core_request_from_tokens()` 调用序列化函数
- **`__init__.py`**：使用 `OmniEngineCoreRequest` 构建阶段间请求
- **`vllm_omni/metrics/`**：`StageRequestMetrics` / `StageStats` 用于性能监控

## 总结

`Orchestrator` 是多阶段推理引擎的"大脑"。它通过两个并行协程实现请求接收和输出路由的解耦，支持 LLM 和 Diffusion 两种阶段类型的统一调度。阶段间转发通过 `process_engine_inputs()` 抽象了输入转换逻辑，支持自定义转换函数。async-chunk 模式通过预热机制实现了流式处理的低延迟启动。整体设计保持了良好的扩展性，新增阶段类型只需实现对应的 stage client 即可接入。
