# `async_omni_engine.py` — 异步多模态引擎代理

## 文件概述

`async_omni_engine.py` 实现了 `AsyncOmniEngine` 类，它是 vLLM-Omni 面向用户的主入口。作为一个轻量级代理（Thin Proxy），它运行在调用者线程中，通过 janus 双向队列与后台的 `Orchestrator` 通信。该文件还负责多阶段流水线的初始化、Stage 配置解析、请求预处理以及资源生命周期管理。

## 关键代码解析

### 1. 请求注入全局 ID

```python
def _inject_global_id(target: Any, request_id: str) -> None:
    if isinstance(target, dict):
        if "additional_information" not in target:
            target["additional_information"] = {}
        if target["additional_information"] is None:
            target["additional_information"] = {}
        if isinstance(target["additional_information"], dict):
            target["additional_information"]["global_request_id"] = [str(request_id)]
```

在发送请求到 Stage-0 之前，将全局请求 ID 注入到 prompt 的 `additional_information` 字段中。这样下游阶段可以通过该 ID 追踪请求来源，尤其在异步分块模式下用于跨阶段关联。

### 2. 升级为 Omni 请求

```python
def _upgrade_to_omni_request(
    request: EngineCoreRequest,
    raw_prompt: Any,
) -> EngineCoreRequest:
    prompt_embeds = request.prompt_embeds
    additional_information = None

    if isinstance(raw_prompt, dict):
        if prompt_embeds is None:
            raw_prompt_embeds = raw_prompt.get("prompt_embeds")
            if isinstance(raw_prompt_embeds, torch.Tensor):
                prompt_embeds = raw_prompt_embeds
        additional_information = serialize_additional_information(
            raw_prompt.get("additional_information"),
            log_prefix="AsyncOmniEngine",
        )

    if prompt_embeds is None and additional_information is None:
        return request

    return OmniEngineCoreRequest(
        request_id=request.request_id,
        # ... 复制所有字段 ...
        prompt_embeds=prompt_embeds,
        additional_information=additional_information,
    )
```

vLLM 原生的 `InputProcessor` 不处理 Omni 特有的 `additional_information` 和 `prompt_embeds` 字段。该函数在 InputProcessor 处理完成后，从原始 prompt 字典中恢复这些字段，将普通 `EngineCoreRequest` 升级为 `OmniEngineCoreRequest`。

### 3. AsyncOmniEngine 初始化流程

```python
class AsyncOmniEngine:
    def __init__(self, model, engine_args=None, stage_init_timeout=300, init_timeout=600, **kwargs):
        self.model = model
        # 1. 解析阶段配置
        self.config_path, self.stage_configs = self._resolve_stage_configs(model, kwargs)

        # 2. 启动 Orchestrator 后台线程
        startup_future = concurrent.futures.Future()
        self.orchestrator_thread = threading.Thread(
            target=self._bootstrap_orchestrator,
            args=(stage_init_timeout, startup_future),
            daemon=True,
            name="orchestrator",
        )
        self.orchestrator_thread.start()

        # 3. 等待初始化完成
        startup_future.result(timeout=startup_timeout)

        # 4. 注册弱引用清理器
        self._weak_finalizer = weakref.finalize(
            self, _weak_shutdown_async_omni_engine,
            self.orchestrator_thread, self.request_queue, self.output_queue, self.rpc_output_queue,
        )
```

初始化分为四个阶段：
1. 解析 YAML 配置或通过 `StageConfigFactory` 自动识别模型流水线
2. 在后台线程中启动 Orchestrator（包括所有 Stage 初始化）
3. 通过 `concurrent.futures.Future` 阻塞等待后台线程初始化完成
4. 注册弱引用终结器，确保 GC 时能正确清理资源

### 4. 阶段初始化 — 并行启动

```python
def _initialize_stages(self, stage_init_timeout):
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, llm_stage_count),
        thread_name_prefix="llm-stage-launch",
    ) as launch_executor:
        for stage_id, stage_cfg in enumerate(self.stage_configs):
            if metadata.stage_type == "diffusion":
                stage_clients[stage_id] = initialize_diffusion_stage(...)
                continue
            llm_launch_futures[stage_id] = launch_executor.submit(
                self._launch_llm_stage, stage_cfg, metadata, ...
            )
        concurrent.futures.wait(list(llm_launch_futures.values()))
```

多个 LLM 阶段通过 `ThreadPoolExecutor` 并行启动，显著缩短多阶段流水线的初始化时间。Diffusion 阶段则同步初始化。每个 LLM 阶段的启动过程包括：设备映射、引擎参数构建、设备锁获取、引擎核心启动。

### 5. 请求添加 — 调用者线程预处理

```python
def add_request(self, request_id, prompt, sampling_params_list=None, ...):
    msg = self._build_add_request_message(
        request_id=request_id,
        prompt=prompt,
        sampling_params_list=sampling_params_list,
        ...
    )
    self.request_queue.sync_q.put_nowait(msg)
```

请求预处理（tokenization、多模态数据处理、Omni 字段恢复）在调用者线程中完成，避免了一次额外的队列往返。处理后的消息通过 janus 队列的同步端发送到 Orchestrator。

### 6. 控制面 RPC

```python
def collective_rpc(self, method, timeout=None, args=(), kwargs=None, stage_ids=None):
    rpc_id = uuid.uuid4().hex
    msg = {
        "type": "collective_rpc",
        "rpc_id": rpc_id,
        "method": method,
        "args": tuple(args),
        "kwargs": kwargs or {},
        "stage_ids": stage_ids,
    }
    with self._rpc_lock:
        self.request_queue.sync_q.put_nowait(msg)
        while True:
            result_msg = self.rpc_output_queue.sync_q.get(timeout=remaining)
            if result_msg.get("rpc_id") == rpc_id:
                return list(result_msg.get("results", []))
```

`collective_rpc` 提供了一个控制面通道，允许用户向特定阶段发送 RPC 调用（如 sleep mode 切换、模型权重更新等）。它使用专用的 `rpc_output_queue` 避免与普通输出队列竞争，并通过 `rpc_id` 匹配响应。

### 7. Diffusion 阶段默认配置

```python
@staticmethod
def _create_default_diffusion_stage_cfg(kwargs):
    # 当没有 YAML 配置时，根据 kwargs 构建默认的 diffusion 阶段配置
    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
        cfg_parallel_size=cfg_parallel_size,
        # ...
    )
    default_stage_cfg = [{
        "stage_id": 0,
        "stage_type": "diffusion",
        "engine_args": { ... },
        "final_output": True,
        "final_output_type": "image",
    }]
```

当用户未提供 YAML 配置且 `StageConfigFactory` 无法识别模型时，自动构建一个单阶段 diffusion 配置作为回退方案。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `AsyncOmniEngine` | 类 | 面向用户的异步引擎代理，管理 Orchestrator 生命周期 |
| `_inject_global_id()` | 函数 | 注入全局请求 ID 到 prompt 附加信息 |
| `_upgrade_to_omni_request()` | 函数 | 将 vLLM 请求升级为 OmniEngineCoreRequest |
| `_weak_shutdown_async_omni_engine()` | 函数 | 弱引用终结器回调，GC 时清理资源 |
| `add_request()` | 方法 | 提交推理请求（同步） |
| `add_request_async()` | 方法 | 提交推理请求（异步） |
| `try_get_output()` | 方法 | 非阻塞获取输出结果 |
| `collective_rpc()` | 方法 | 发送控制面 RPC 到指定阶段 |
| `shutdown()` | 方法 | 关闭引擎和所有阶段 |
| `_initialize_stages()` | 方法 | 并行初始化所有阶段 |
| `_resolve_stage_configs()` | 方法 | 解析阶段配置（YAML / StageConfigFactory / 默认） |

## 与其他模块的关系

- **`orchestrator.py`**：`AsyncOmniEngine` 在后台线程中创建并运行 `Orchestrator`
- **`stage_init.py`**：所有阶段初始化辅助函数（设备映射、引擎参数构建等）来自此模块
- **`stage_engine_core_client.py`**：每个 LLM 阶段对应一个 `StageEngineCoreClient`
- **`output_processor.py`**：每个阶段有一个 `MultimodalOutputProcessor` 实例
- **`serialization.py`**：`_upgrade_to_omni_request()` 调用 `serialize_additional_information()`
- **`__init__.py`**：使用 `OmniEngineCoreRequest` 数据结构
- **`vllm_omni/config/stage_config.py`**：`StageConfigFactory` 用于自动识别模型流水线
- **`vllm_omni/entrypoints/utils.py`**：`load_stage_configs_from_yaml()` 加载 YAML 配置

## 总结

`AsyncOmniEngine` 是 vLLM-Omni 的核心入口类。它采用"调用者线程做预处理、后台线程做调度"的设计模式，通过 janus 双向队列实现线程间通信。初始化阶段支持并行启动多个 LLM 引擎以加速启动，运行时通过三条队列（请求、输出、RPC）实现完全解耦的数据面和控制面通信。该设计使得用户可以在自己的异步事件循环中无缝集成多阶段推理能力。
