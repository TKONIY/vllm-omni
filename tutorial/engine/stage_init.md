# `stage_init.py` — 阶段初始化辅助工具集

## 文件概述

`stage_init.py` 提取了多阶段引擎初始化过程中的所有辅助逻辑，包括：配置元数据提取、环境准备、设备映射、引擎参数构建、设备锁管理、Diffusion 阶段初始化、以及初始化失败时的清理。这些函数被 `AsyncOmniEngine._initialize_stages()` 调用，实现了关注点分离。

## 关键代码解析

### 1. StageMetadata — 阶段元数据

```python
@dataclass
class StageMetadata:
    stage_id: int
    stage_type: Literal["llm", "diffusion"]
    engine_output_type: str | None           # "audio"/"image"/"latent"
    is_comprehension: bool                   # 是否为理解阶段
    requires_multimodal_data: bool           # 是否需要多模态输入数据
    engine_input_source: list[int]           # 上游阶段 ID 列表
    final_output: bool                       # 是否产出最终用户输出
    final_output_type: str | None            # 最终输出类型
    default_sampling_params: OmniSamplingParams  # 默认采样参数
    custom_process_input_func: Callable | None   # 自定义输入处理函数
    model_stage: str | None                  # "thinker"/"talker"
    runtime_cfg: Any                         # 运行时配置（设备、batch size 等）
```

`StageMetadata` 是从 stage config 对象中提取的纯数据结构，包含一个阶段运行所需的全部元数据。它被 `StageEngineCoreClient` 和 `Orchestrator` 使用。

### 2. StartedLlmStage — 已启动阶段资源

```python
@dataclass
class StartedLlmStage:
    stage_id: int
    metadata: Any
    vllm_config: Any
    executor_class: type
    engine_manager: Any      # 引擎进程管理器
    coordinator: Any         # 协调器
    addresses: Any           # ZMQ 地址
```

表示一个已经完成 EngineCore 启动但尚未 attach 到客户端的 LLM 阶段。这个中间状态允许并行启动多个阶段，然后再依次 attach。

### 3. 元数据提取

```python
def extract_stage_metadata(stage_config: Any) -> StageMetadata:
    stage_id = stage_config.stage_id
    stage_type = getattr(stage_config, "stage_type", "llm")
    engine_args = stage_config.engine_args

    # 解析默认采样参数
    default_sp = _to_dict(getattr(stage_config, "default_sampling_params", {}))
    SPClass = SamplingParams if stage_type == "llm" else OmniDiffusionSamplingParams
    default_sampling_params = SPClass(**default_sp)

    # 解析自定义输入处理函数（动态导入）
    custom_process_input_func = None
    if hasattr(stage_config, "custom_process_input_func"):
        mod_path, fn_name = stage_config.custom_process_input_func.rsplit(".", 1)
        custom_process_input_func = getattr(importlib.import_module(mod_path), fn_name)

    return StageMetadata(
        stage_id=stage_id,
        stage_type=stage_type,
        # ...
    )
```

从 OmegaConf/SimpleNamespace 格式的 stage config 中提取所有元数据。`custom_process_input_func` 通过 Python 动态导入机制加载（`"module.path.function_name"` 格式）。

### 4. 环境准备

```python
def prepare_engine_environment() -> None:
    from vllm_omni.plugins import load_omni_general_plugins
    load_omni_general_plugins()

    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
```

一次性全局设置：
- 加载 Omni 插件（平台适配等）
- 强制使用 `spawn` 进程启动方式（避免 CUDA fork 问题）

### 5. 设备映射

```python
def setup_stage_devices(stage_id: int, runtime_cfg: Any) -> None:
    from vllm_omni.platforms import current_omni_platform
    device_type = current_omni_platform.device_type
    set_stage_devices(
        stage_id,
        runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else None,
        device_type=device_type,
    )
```

为每个阶段设置可见设备。调用 `set_stage_devices()` 修改 `CUDA_VISIBLE_DEVICES`（或对应平台的设备控制环境变量），确保不同阶段使用不同的 GPU。

### 6. 引擎参数构建

```python
def build_engine_args_dict(stage_config, model, stage_connector_spec=None):
    engine_args_dict = _to_dict(engine_args)
    engine_args_dict["model"] = model
    engine_args_dict["stage_id"] = stage_id
    if engine_args_dict.get("async_chunk", False):
        engine_args_dict["stage_connector_spec"] = dict(stage_connector_spec or {})
    if stage_type != "diffusion":
        resolve_worker_cls(engine_args_dict)
    return engine_args_dict

def build_vllm_config(stage_config, model, stage_connector_spec=None, engine_args_dict=None):
    if engine_args_dict is None:
        engine_args_dict = build_engine_args_dict(...)
    filtered_engine_args_dict = filter_dataclass_kwargs(OmniEngineArgs, engine_args_dict)
    omni_engine_args = OmniEngineArgs(**filtered_engine_args_dict)
    vllm_config = omni_engine_args.create_engine_config(usage_context=UsageContext.LLM_CLASS)
    executor_class = Executor.get_class(vllm_config)
    return vllm_config, executor_class
```

两步流程：
1. `build_engine_args_dict()` 构建原始参数字典（注入 model、stage_id、worker_cls）
2. `build_vllm_config()` 过滤参数、创建 `OmniEngineArgs`、生成 `VllmConfig` 和对应的 `Executor` 类

### 7. 设备锁管理

```python
def acquire_device_locks(stage_id, engine_args_dict, stage_init_timeout=300):
    # 计算需要锁定的设备数
    num_devices_per_stage = (
        tensor_parallel_size * pipeline_parallel_size *
        data_parallel_size * prefill_context_parallel_size *
        sequence_parallel_size * cfg_parallel_size
    )

    # 获取物理设备 ID
    visible_devices_str = os.environ.get(device_control_env)
    physical_devices = [int(x) for x in visible_devices_str.split(",")]

    # 通过 fcntl.flock 获取排他锁
    for device_id in devices_to_lock:
        lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fds.append(lock_fd)
    return lock_fds

def release_device_locks(lock_fds):
    for lock_fd in lock_fds:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
```

设备锁机制防止多个阶段同时初始化同一 GPU（CUDA 初始化非线程安全）。使用文件系统级别的 `fcntl.flock` 排他锁，锁文件路径基于物理设备 ID。超时后会记录警告并继续（避免死锁）。

### 8. Diffusion 阶段初始化

```python
def initialize_diffusion_stage(model, stage_cfg, metadata):
    od_config = OmniDiffusionConfig.from_kwargs(
        model=model,
        **_to_dict(stage_cfg.engine_args),
    )
    return StageDiffusionClient(model, od_config, metadata)
```

Diffusion 阶段不使用 vLLM 的 EngineCore，而是通过 `StageDiffusionClient` 直接管理。配置从 stage config 的 engine_args 中提取。

### 9. 初始化完成与失败处理

```python
def finalize_initialized_stages(stage_clients, input_processor):
    if any(client is None for client in stage_clients):
        raise RuntimeError("Stage initialization completed with missing stage clients")
    initialized_stage_clients = [c for c in stage_clients if c is not None]
    default_sampling_params_list = [c.default_sampling_params for c in initialized_stage_clients]
    stage_metadata = [{"final_output": c.final_output, ...} for c in initialized_stage_clients]
    return initialized_stage_clients, default_sampling_params_list, stage_metadata

def cleanup_failed_stage_initialization(stage_clients, started_llm_stages):
    # 反向关闭已初始化的 stage client
    for cleanup_stage_id, stage_client in reversed(list(enumerate(stage_clients))):
        if stage_client is not None:
            stage_client.shutdown()
    # 关闭已启动但未 attach 的引擎进程
    for started in reversed(started_llm_stages):
        if stage_clients[started.stage_id] is None:
            close_started_llm_stage(started)
```

- `finalize_initialized_stages()` 验证所有阶段都已成功初始化，并构建运行时元数据
- `cleanup_failed_stage_initialization()` 在初始化失败时反向清理所有资源

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `StageMetadata` | 数据类 | 阶段元数据容器 |
| `StartedLlmStage` | 数据类 | 已启动但未 attach 的 LLM 阶段资源 |
| `extract_stage_metadata()` | 函数 | 从 stage config 提取元数据 |
| `prepare_engine_environment()` | 函数 | 全局环境准备（插件、spawn 方式） |
| `setup_stage_devices()` | 函数 | 设置阶段可见设备 |
| `build_engine_args_dict()` | 函数 | 构建引擎参数字典 |
| `build_vllm_config()` | 函数 | 生成 VllmConfig 和 Executor 类 |
| `acquire_device_locks()` | 函数 | 获取设备排他锁 |
| `release_device_locks()` | 函数 | 释放设备锁 |
| `load_omni_transfer_config_for_model()` | 函数 | 加载阶段间传输配置 |
| `get_stage_connector_spec()` | 函数 | 获取阶段连接器规格 |
| `initialize_diffusion_stage()` | 函数 | 初始化 Diffusion 阶段 |
| `finalize_initialized_stages()` | 函数 | 验证初始化并构建运行时元数据 |
| `cleanup_failed_stage_initialization()` | 函数 | 初始化失败时清理资源 |
| `close_started_llm_stage()` | 函数 | 关闭已启动但未 attach 的阶段 |

## 与其他模块的关系

- **`async_omni_engine.py`**：`_initialize_stages()` 和 `_launch_llm_stage()` 大量使用本文件的函数
- **`arg_utils.py`**：`build_vllm_config()` 创建 `OmniEngineArgs` 并调用 `create_engine_config()`
- **`worker_cls_utils.py`**：`build_engine_args_dict()` 调用 `resolve_worker_cls()`
- **`stage_engine_core_client.py`**：使用 `StageMetadata` 初始化客户端
- **`vllm_omni/diffusion/`**：Diffusion 阶段使用 `OmniDiffusionConfig` 和 `StageDiffusionClient`
- **`vllm_omni/distributed/omni_connectors/`**：阶段间传输配置和连接器规格

## 总结

`stage_init.py` 是多阶段初始化的"瑞士军刀"，将复杂的初始化过程分解为可测试、可复用的独立函数。设备锁机制确保了并行初始化的安全性，两阶段启动模式（launch + attach）实现了最大程度的并行化。完善的错误处理和资源清理逻辑保证了即使部分阶段初始化失败，也不会泄漏进程或设备资源。
