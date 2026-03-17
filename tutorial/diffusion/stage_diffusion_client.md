# `stage_diffusion_client.py` — 阶段扩散客户端

## 文件概述

`stage_diffusion_client.py` 实现了 `StageDiffusionClient`，用于在 vLLM-Omni V1 的多阶段（Stage）架构中封装 `AsyncOmniDiffusion` 引擎。它向编排器（Orchestrator）暴露与 `StageEngineCoreClient` 相同的异步接口，使扩散引擎可以作为流水线中的一个阶段运行。

## 关键代码解析

### StageDiffusionClient 初始化

```python
class StageDiffusionClient:
    stage_type: str = "diffusion"

    def __init__(self, model: str, od_config: OmniDiffusionConfig, metadata: StageMetadata):
        self.stage_id = metadata.stage_id
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self._engine = AsyncOmniDiffusion(model=model, od_config=od_config)
        self._output_queue: asyncio.Queue[OmniRequestOutput] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task] = {}
```

从 `StageMetadata` 获取阶段配置（阶段 ID、输出类型等），创建异步扩散引擎和输出队列。

### 异步请求处理

```python
async def add_request_async(self, request_id, prompt, sampling_params):
    task = asyncio.create_task(self._run(request_id, prompt, sampling_params))
    self._tasks[request_id] = task

async def _run(self, request_id, prompt, sampling_params):
    try:
        result = await self._engine.generate(prompt, sampling_params, request_id)
        await self._output_queue.put(result)
    finally:
        self._tasks.pop(request_id, None)

def get_diffusion_output_async(self) -> OmniRequestOutput | None:
    try:
        return self._output_queue.get_nowait()
    except asyncio.QueueEmpty:
        return None
```

每个请求创建独立的 asyncio Task，结果放入异步队列。Orchestrator 通过非阻塞的 `get_diffusion_output_async` 拉取结果。

### RPC 透传

```python
async def collective_rpc_async(self, method, timeout=None, args=(), kwargs=None):
    if method in {"add_lora", "remove_lora", "start_profile", "stop_profile", ...}:
        target = getattr(self._engine, method, None)
        result = target(*args, **kwargs)
        return await result
    if method in {"sleep", "wake_up"}:
        return await loop.run_in_executor(...)
    return {"supported": False, "todo": True, "reason": ...}
```

将控制类 RPC（LoRA 管理、性能分析、休眠唤醒）透传给底层引擎，未实现的方法返回 `todo` 标记。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `StageDiffusionClient` | 类 | 阶段客户端，封装 AsyncOmniDiffusion 供 Orchestrator 使用 |
| `add_request_async` | 异步方法 | 提交扩散请求，创建 asyncio Task |
| `get_diffusion_output_async` | 方法 | 非阻塞拉取扩散结果 |
| `collective_rpc_async` | 异步方法 | 控制类 RPC 透传 |
| `abort_requests_async` | 异步方法 | 取消正在进行的请求 |

## 与其他模块的关系

- 封装 `entrypoints/async_omni_diffusion.py` 中的 `AsyncOmniDiffusion`
- 被 `engine/` 中的 Orchestrator 作为 Stage 客户端使用
- 接收 `StageMetadata` 配置来自 `engine/stage_init.py`
- 输出 `OmniRequestOutput` 与其他 Stage 的输出格式统一

## 总结

`StageDiffusionClient` 是扩散引擎与多阶段编排架构之间的适配层。它将同步的扩散推理封装为异步接口，通过 asyncio Task 和输出队列实现非阻塞的请求处理。这使得扩散模型可以作为多模态流水线中的一个独立阶段运行。
