# `async_omni.py` — 异步多阶段管线编排器

## 文件概述

`AsyncOmni` 是 vLLM-Omni 在线服务的核心入口类。它同时实现了 vLLM 的 `EngineClient` 接口和自定义的 `OmniBase` 基类，通过 `AsyncOmniEngine` 编排多阶段推理管线（如文本理解 + 语音合成）。该类是 OpenAI API 服务器调用的主引擎客户端。

## 关键代码解析

### 类定义与初始化

```python
class AsyncOmni(EngineClient, OmniBase):
    def __init__(self, model: str, **kwargs: Any) -> None:
        OmniBase.__init__(self, model=model, **kwargs)
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False
        self.final_output_task: asyncio.Task | None = None
        # ...
        stage_index = self._get_comprehension_stage_index()
        if stage_index is None:
            self.io_processor = None
        else:
            vllm_config = self.engine.stage_vllm_configs[stage_index]
            self.io_processor = get_io_processor(vllm_config, io_processor_plugin)
```

初始化过程中，`AsyncOmni` 通过 `OmniBase` 构建 `AsyncOmniEngine`，然后查找"理解阶段"（comprehension stage）以获取 tokenizer 和 IO 处理器。支持暂停/恢复生成的条件变量机制。

### 核心生成方法

```python
async def generate(
    self,
    prompt: OmniPromptType,
    request_id: str,
    sampling_params_list: Sequence[OmniSamplingParams] | None = None,
    *,
    output_modalities: list[str] | None = None,
) -> AsyncGenerator[OmniRequestOutput, None]:
```

`generate()` 是异步生成器方法，工作流程：
1. 等待暂停条件释放
2. 启动后台输出分发任务（`_final_output_handler`）
3. 将请求提交到 `AsyncOmniEngine`
4. 从编排器队列读取结果并逐个 yield `OmniRequestOutput`

### 后台输出分发

```python
def _final_output_handler(self) -> None:
    async def _final_output_loop():
        while True:
            msg = await engine.try_get_output_async()
            if msg is None:
                await asyncio.sleep(_FINAL_OUTPUT_IDLE_SLEEP_S)
                continue
            should_continue, _, stage_id, req_state = self._handle_output_message(msg)
            if should_continue:
                continue
            await req_state.queue.put(msg)
    self.final_output_task = asyncio.create_task(_final_output_loop())
```

后台 coroutine 从编排器输出队列读取消息，按 `request_id` 路由到各请求的独立 `asyncio.Queue`，实现请求间的隔离。

### 控制面方法

```python
async def collective_rpc(self, method, timeout, args, kwargs, stage_ids):
    results = await self.engine.collective_rpc_async(...)
    # 对不支持的阶段返回 TODO 标记，而非抛出异常
```

通过 `collective_rpc` 将控制命令（如 `start_profile`、`sleep`、`add_lora`）广播到所有阶段引擎。对尚未实现的阶段采用 best-effort 策略。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `AsyncOmni` | 类 | 异步多阶段推理编排器，实现 EngineClient |
| `generate()` | 异步生成器 | 提交请求并流式返回多阶段输出 |
| `_final_output_handler()` | 方法 | 启动后台输出分发任务 |
| `_process_orchestrator_results()` | 异步生成器 | 从请求队列读取并构造输出 |
| `collective_rpc()` | 异步方法 | 向阶段引擎广播 RPC 调用 |
| `abort()` | 异步方法 | 中止请求 |
| `pause_generation()` / `resume_generation()` | 异步方法 | 暂停/恢复生成 |
| `get_tokenizer()` | 异步方法 | 获取理解阶段的 tokenizer |
| `shutdown()` | 方法 | 关闭引擎和后台任务 |

## 与其他模块的关系

- 继承 `OmniBase`（`omni_base.py`）获取引擎初始化和共享逻辑
- 实现 vLLM 的 `EngineClient` 接口，可被 `OpenAIServingChat` 等服务层直接使用
- 使用 `ClientRequestState`（`client_request_state.py`）跟踪每个请求的状态和队列
- 底层由 `AsyncOmniEngine`（`vllm_omni.engine`）驱动实际的多阶段编排
- 被 `openai/api_server.py` 中的 `build_async_omni()` 创建并注入到 FastAPI 应用状态

## 总结

`AsyncOmni` 是 vLLM-Omni 在线服务的心脏，将多阶段推理管线封装为标准的 `EngineClient` 接口。通过后台任务分发机制和按请求隔离的队列，实现了高效的异步流式输出。它同时提供了完整的控制面（暂停、性能分析、LoRA 管理等），是连接 HTTP API 层和底层引擎的关键桥梁。
