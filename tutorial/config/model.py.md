# `config/model.py` — OmniModelConfig 模型配置

## 文件概述

`model.py` 定义了 `OmniModelConfig`，扩展 vLLM 的 `ModelConfig`，添加多阶段流水线处理所需的配置字段。每个流水线阶段会创建一个 `OmniModelConfig` 实例。

## 关键代码解析

### 核心字段

```python
@config(config=ConfigDict(arbitrary_types_allowed=True))
class OmniModelConfig(ModelConfig):
    stage_id: int = 0
    async_chunk: bool = False
    model_stage: str = "thinker"
    model_arch: str | None = None
    worker_type: str | None = None
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    stage_connector_config: dict[str, Any] = field(
        default_factory=lambda: {"name": "SharedMemoryConnector", "extra": {}}
    )
    omni_kv_config: dict | None = None
    codec_frame_rate_hz: float | None = None
    task_type: str | None = None
```

关键字段说明：
- `stage_id`：阶段编号，标识流水线中的位置
- `model_stage`：阶段角色，如 `"thinker"`（理解）或 `"talker"`（生成）
- `model_arch`：模型架构名称，用于覆盖 HF 配置中的架构列表
- `worker_type`：工作器类型，`"ar"`（自回归）或 `"generation"`（生成）
- `stage_connector_config`：阶段间连接器配置

### 模型注册表覆盖

```python
@property
def registry(self):
    return me_models.OmniModelRegistry
```

使用 omni 专用的模型注册表，而非 vLLM 默认注册表。

### 多阶段文本配置提取

```python
def draw_hf_text_config(self):
    if self.hf_config_name is None:
        return get_hf_text_config(self.hf_config)
    try:
        stage_config = getattr(self.hf_config, self.hf_config_name)
        return stage_config.get_text_config()
    except AttributeError:
        return get_hf_text_config(self.hf_config)
```

对于多阶段模型（如 Qwen2.5-Omni），`hf_config` 包含 `thinker_config` 和 `talker_config` 等子配置。此方法根据 `hf_config_name` 提取对应阶段的文本配置。

### 初始化后处理

```python
def __post_init__(self, ...):
    super().__post_init__(...)

    # Qwen3-TTS: 自动推断编解码器帧率
    if self.codec_frame_rate_hz is None and self.model_arch == "Qwen3TTSTalkerForConditionalGenerationARVLLM":
        talker_cfg = getattr(self.hf_config, "talker_config", None)
        # ... 从 position_id_per_seconds 推断帧率

    # 覆盖 hf_text_config 并重新计算依赖属性
    new_hf_text_config = self.draw_hf_text_config()
    if new_hf_text_config is not self.hf_text_config:
        self.hf_text_config = new_hf_text_config
        self.max_model_len = self.get_and_verify_max_len(self.original_max_model_len)
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniModelConfig` | 数据类 | 多阶段模型配置 |
| `registry` | 属性 | 返回 Omni 模型注册表 |
| `architectures` | 属性 | 返回模型架构列表（可被 model_arch 覆盖） |
| `embedding_size` | 属性 | 返回嵌入维度（支持阶段覆盖） |
| `draw_hf_text_config` | 方法 | 提取阶段对应的 HF 文本配置 |

## 与其他模块的关系

- 继承 `vllm.config.ModelConfig`
- 被 `stage_config.py` 中的 `StageConfig.to_omegaconf()` 使用配置字段
- 引用 `model_executor.models.OmniModelRegistry` 作为模型注册表
- 被引擎层用于创建每个阶段的模型配置

## 总结

`OmniModelConfig` 是连接流水线配置与模型执行的桥梁，通过灵活的字段设计支持多种模型架构（Qwen-Omni、Bagel、GLM-Image、CosyVoice 等）的多阶段部署。
