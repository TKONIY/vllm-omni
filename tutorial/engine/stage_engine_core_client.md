# `stage_engine_core_client.py` — 单阶段引擎核心客户端

## 文件概述

`stage_engine_core_client.py` 实现了 `StageEngineCoreClient` 类，它是 vLLM-Omni 中每个 LLM 阶段的引擎客户端。通过直接继承 vLLM 的 `AsyncMPClient`，它完全复用了 vLLM 的 ZMQ 通信、引擎核心进程管理、输出队列等基础设施，同时增加了阶段特有的元数据和输入处理能力。

## 关键代码解析

### 1. 初始化 — 存储元数据并启动引擎

```python
class StageEngineCoreClient(AsyncMPClient):
    def __init__(self, vllm_config, executor_class, metadata: StageMetadata,
                 client_addresses=None, engine_manager=None, coordinator=None):
        # 存储阶段元数据
        self.stage_id = metadata.stage_id
        self.stage_type = metadata.stage_type              # "llm" 或 "diffusion"
        self.engine_output_type = metadata.engine_output_type  # "audio"/"image"/"latent"
        self.is_comprehension = metadata.is_comprehension  # 是否为理解阶段
        self.requires_multimodal_data = metadata.requires_multimodal_data
        self.engine_input_source = metadata.engine_input_source  # 上游阶段 ID 列表
        self.final_output = metadata.final_output          # 是否产出最终输出
        self.final_output_type = metadata.final_output_type
        self.default_sampling_params = metadata.default_sampling_params
        self.custom_process_input_func = metadata.custom_process_input_func
        self.model_stage = metadata.model_stage            # "thinker"/"talker"

        self.engine_outputs: Any = None  # 存储上游输出，供下游读取

        # 调用 AsyncMPClient.__init__，建立 ZMQ 连接
        super().__init__(
            vllm_config, executor_class,
            log_stats=False,
            client_addresses=client_addresses,
        )
        # 接管 engine_manager 和 coordinator 的所有权
        if engine_manager is not None:
            self.resources.engine_manager = engine_manager
        if coordinator is not None:
            self.resources.coordinator = coordinator
```

初始化分为三步：
1. 从 `StageMetadata` 中提取并存储所有阶段属性
2. 调用 `AsyncMPClient.__init__()` 建立与 EngineCore 进程的 ZMQ 通信通道
3. 将外部传入的 engine_manager/coordinator 注入到 resources 中（所有权转移）

如果初始化失败，会尝试调用 `shutdown()` 清理已创建的资源。

### 2. 请求提交（覆写）

```python
async def add_request_async(self, request: EngineCoreRequest) -> None:
    logger.info(f"[StageEngineCoreClient] Stage-{self.stage_id} adding request: {request.request_id}")
    await super().add_request_async(request)
```

简单包装基类方法，添加了调试日志。实际的请求序列化和 ZMQ 发送由 `AsyncMPClient` 完成。

### 3. 上游输出存储

```python
def set_engine_outputs(self, engine_outputs: EngineCoreOutput) -> None:
    self.engine_outputs = engine_outputs
```

Orchestrator 在阶段间转发时，先将当前阶段的输出存储到 `engine_outputs` 字段上。下游阶段的 `process_engine_inputs()` 会从上游 client 的 `engine_outputs` 中读取数据。

### 4. 输入处理 — process_engine_inputs

```python
def process_engine_inputs(self, stage_list, prompt=None):
    from vllm_omni.inputs.data import OmniTokensPrompt

    # 优先使用自定义处理函数
    if self.custom_process_input_func is not None:
        return self.custom_process_input_func(
            stage_list,
            self.engine_input_source,
            prompt,
            self.requires_multimodal_data,
        )

    # 默认逻辑：从上游阶段读取 token IDs
    source_id = self.engine_input_source[0]
    source_outputs = stage_list[source_id].engine_outputs

    if not isinstance(prompt, list):
        prompt = [prompt]

    mm_data = {
        so.request_id: p.get("multi_modal_data")
        for so, p in zip(source_outputs, prompt)
    }

    return [
        OmniTokensPrompt(
            prompt_token_ids=so.outputs[0].token_ids,
            multi_modal_data=(mm_data[so.request_id] if self.requires_multimodal_data else None),
        )
        for so in source_outputs
    ]
```

输入处理逻辑：
- **自定义函数**：如果配置了 `custom_process_input_func`（在 stage config 中指定），将全部 stage 列表和原始 prompt 传给它
- **默认逻辑**：从 `engine_input_source` 指定的上游阶段读取 token IDs，可选地携带多模态数据

### 5. 控制面 RPC 转发

```python
async def collective_rpc_async(self, method, timeout=None, args=(), kwargs=None):
    return await super().collective_rpc_async(
        method=method, timeout=timeout, args=args, kwargs=kwargs,
    )
```

将控制面 RPC（如 sleep mode 切换）直接转发给底层的 `AsyncMPClient`，再由其扇出到所有 worker 进程。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `StageEngineCoreClient` | 类 | 单阶段 LLM 引擎客户端，继承 vLLM AsyncMPClient |
| `__init__()` | 方法 | 存储阶段元数据，建立 ZMQ 通信 |
| `add_request_async()` | 方法 | 向阶段引擎提交请求（覆写添加日志） |
| `set_engine_outputs()` | 方法 | 存储当前阶段输出，供下游阶段读取 |
| `process_engine_inputs()` | 方法 | 处理上游阶段输出，生成当前阶段的输入 |
| `collective_rpc_async()` | 方法 | 转发控制面 RPC 到引擎核心 |

## 与其他模块的关系

- **`orchestrator.py`**：Orchestrator 持有 `StageEngineCoreClient` 列表，调用其 `add_request_async()`、`get_output_async()`、`set_engine_outputs()` 和 `process_engine_inputs()`
- **`stage_init.py`**：`StageMetadata` 数据类提供初始化所需的元数据；`build_vllm_config()` 构建 `vllm_config`
- **`async_omni_engine.py`**：在 `_attach_llm_stage()` 中创建 `StageEngineCoreClient` 实例
- **vLLM `AsyncMPClient`**：提供 ZMQ 通信、输出队列、请求序列化等全部基础设施
- **`vllm_omni/inputs/data.py`**：`OmniTokensPrompt` 用于表示阶段间传递的 token 数据

## 总结

`StageEngineCoreClient` 体现了"继承复用"的设计理念。通过继承 vLLM 的 `AsyncMPClient`，它获得了完整的引擎核心通信能力（ZMQ、进程管理、输出队列），只需在此基础上增加阶段元数据管理和输入处理逻辑。`process_engine_inputs()` 方法支持自定义处理函数，使得不同模型的阶段间数据转换可以灵活定制。`engine_outputs` 字段作为阶段间数据传递的临时缓冲区，由 Orchestrator 在转发时协调读写。
