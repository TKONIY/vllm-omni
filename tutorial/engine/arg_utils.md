# `arg_utils.py` — 多阶段引擎参数配置

## 文件概述

`arg_utils.py` 定义了 vLLM-Omni 的引擎参数类 `OmniEngineArgs` 和 `AsyncOmniEngineArgs`，分别继承自 vLLM 的 `EngineArgs` 和 `AsyncEngineArgs`。这两个类在 vLLM 基础参数之上添加了多阶段流水线所需的配置字段（如阶段 ID、模型阶段类型、输出类型等），并重写了 `create_model_config()` 方法以生成 `OmniModelConfig` 配置对象。

## 关键代码解析

### 1. Omni HF 配置注册

```python
def _register_omni_hf_configs() -> None:
    try:
        from transformers import AutoConfig
        from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    except Exception as exc:
        logger.warning("Skipping omni HF config registration due to import error: %s", exc)
        return
    try:
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoConfig.register("cosyvoice3", CosyVoice3Config)
    except ValueError:
        return
```

该函数向 HuggingFace `AutoConfig` 注册 vLLM-Omni 自定义模型的配置类。注册后，`AutoConfig.from_pretrained()` 就能正确加载 Qwen3-TTS 和 CosyVoice3 等模型的配置。使用 best-effort 策略，注册失败不会中断程序。

### 2. Omni 模型注册

```python
def register_omni_models_to_vllm():
    from vllm.model_executor.models import ModelRegistry
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    _register_omni_hf_configs()
    supported_archs = ModelRegistry.get_supported_archs()
    for arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items():
        if arch not in supported_archs:
            ModelRegistry.register_model(
                arch,
                f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}:{cls_name}"
            )
```

将 vLLM-Omni 的自定义模型架构注册到 vLLM 的 `ModelRegistry` 中，避免模型加载时找不到架构的错误。仅注册尚未注册的架构。

### 3. OmniEngineArgs 核心字段

```python
@dataclass
class OmniEngineArgs(EngineArgs):
    stage_id: int = 0                                    # 阶段编号
    model_stage: str = "thinker"                         # 阶段类型 (thinker/talker)
    model_arch: str | None = None                        # 模型架构名
    engine_output_type: str | None = None                # 输出类型 (audio/image/latents)
    hf_config_name: str | None = None                    # HF 配置名
    custom_process_next_stage_input_func: str | None = None  # 自定义输入处理函数路径
    stage_connector_spec: dict[str, Any] = field(default_factory=dict)  # 阶段连接器配置
    async_chunk: bool = False                            # 是否启用异步分块
    omni_kv_config: dict | None = None                   # KV 缓存配置
    worker_type: str | None = None                       # Worker 类型 (ar/generation)
    task_type: str | None = None                         # TTS 任务类型
```

每个字段都服务于多阶段推理流水线的特定需求。例如 `model_stage` 区分 "thinker"（理解/推理阶段）和 "talker"（语音生成阶段）；`async_chunk` 控制是否使用共享内存进行流式传输。

### 4. create_model_config() 方法

```python
def create_model_config(self) -> OmniModelConfig:
    # GGUF 格式特殊处理
    if is_gguf(self.model):
        self.quantization = self.load_format = "gguf"

    # 注册 Omni 模型
    self._ensure_omni_models_registered()

    # 构建 stage_connector_config
    stage_connector_config = {
        "name": self.stage_connector_spec.get("name", "SharedMemoryConnector"),
        "extra": self.stage_connector_spec.get("extra", {}).copy(),
    }
    stage_connector_config["extra"]["stage_id"] = self.stage_id

    # 创建 OmniModelConfig（包含所有 vLLM 基础字段 + Omni 扩展字段）
    omni_config = OmniModelConfig(
        model=self.model,
        # ... 基础字段 ...
        # Omni 扩展字段
        stage_id=self.stage_id,
        async_chunk=self.async_chunk,
        model_stage=self.model_stage,
        stage_connector_config=stage_connector_config,
        # ...
    )
    omni_config.hf_config.architectures = omni_config.architectures
    return omni_config
```

该方法是整个参数系统的核心输出点。它将所有引擎参数转化为一个 `OmniModelConfig` 对象，供后续的引擎初始化和模型加载使用。注意 `stage_connector_config` 的构建逻辑：默认使用 `SharedMemoryConnector`，并自动注入 `stage_id`。

### 5. __post_init__ 中的插件加载

```python
def __post_init__(self) -> None:
    load_omni_general_plugins()
    super().__post_init__()
```

在 dataclass 初始化完成后，首先加载 Omni 通用插件（如自定义平台支持），然后调用 vLLM 基类的初始化逻辑。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `_register_omni_hf_configs()` | 函数 | 向 HuggingFace AutoConfig 注册自定义模型配置 |
| `register_omni_models_to_vllm()` | 函数 | 将 Omni 模型架构注册到 vLLM ModelRegistry |
| `OmniEngineArgs` | 数据类 | 同步引擎参数，扩展 vLLM EngineArgs 添加多阶段字段 |
| `AsyncOmniEngineArgs` | 数据类 | 异步引擎参数，扩展 vLLM AsyncEngineArgs 添加多阶段字段 |
| `create_model_config()` | 方法 | 将引擎参数转换为 OmniModelConfig 配置对象 |

## 与其他模块的关系

- **`vllm_omni/config.py`**：`create_model_config()` 产出 `OmniModelConfig` 对象
- **`stage_init.py`**：`build_vllm_config()` 调用 `OmniEngineArgs` 来构建每个阶段的配置
- **`worker_cls_utils.py`**：在 `build_engine_args_dict()` 中通过 `resolve_worker_cls()` 解析 worker 类
- **`vllm_omni/plugins.py`**：`__post_init__` 调用 `load_omni_general_plugins()` 加载平台插件
- **`vllm_omni/model_executor/models/registry.py`**：提供 `_OMNI_MODELS` 模型注册表

## 总结

`arg_utils.py` 是 vLLM-Omni 引擎参数系统的核心。`OmniEngineArgs` 和 `AsyncOmniEngineArgs` 两个数据类通过继承 vLLM 原生参数类，在保持完全兼容的同时，添加了多阶段流水线所需的所有配置字段。`create_model_config()` 方法是参数到配置的转换枢纽，确保模型注册、设备映射、连接器配置等全部正确设置。两个参数类的实现几乎一致（`OmniEngineArgs` 用于同步场景，`AsyncOmniEngineArgs` 用于异步场景），后续版本可能会考虑合并。
