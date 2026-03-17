# `config/stage_config.py` — 流水线阶段配置系统

## 文件概述

`stage_config.py` 实现了 vllm-omni 的流水线配置系统，包括阶段配置（`StageConfig`）、流水线定义（`ModelPipeline`）和配置工厂（`StageConfigFactory`）。流水线结构由模型开发者通过 YAML 文件定义，运行时参数通过 CLI 覆盖。

## 关键代码解析

### StageType — 阶段类型枚举

```python
class StageType(str, Enum):
    LLM = "llm"
    DIFFUSION = "diffusion"
```

两种阶段类型：
- `LLM`：自回归语言模型阶段
- `DIFFUSION`：扩散模型阶段（图像/音频生成）

### StageConfig — 单阶段配置

```python
@dataclass
class StageConfig:
    stage_id: int                         # 阶段编号
    model_stage: str                      # 阶段角色 (thinker/talker)
    stage_type: StageType = StageType.LLM # 阶段类型
    input_sources: list[int]              # 输入来源阶段 ID
    final_output: bool = False            # 是否为最终输出阶段
    final_output_type: str | None = None  # 输出类型 (text/audio/image)
    worker_type: str | None = None        # 工作器类型 (ar/generation)
    scheduler_cls: str | None = None      # 自定义调度器类名
    yaml_engine_args: dict                # YAML 中的引擎参数
    yaml_runtime: dict                    # YAML 中的运行时参数
    yaml_extras: dict                     # YAML 透传字段
    runtime_overrides: dict               # CLI 覆盖参数
```

`to_omegaconf()` 方法将配置转换为 OmegaConf 格式，实现与遗留系统的兼容：

```python
def to_omegaconf(self):
    engine_args = dict(self.yaml_engine_args)
    engine_args["model_stage"] = self.model_stage
    # CLI 覆盖优先
    for key, value in self.runtime_overrides.items():
        if key not in ("devices", "max_batch_size"):
            engine_args[key] = value
    # ...
    return create_config(config_dict)
```

### ModelPipeline — 流水线定义

```python
@dataclass
class ModelPipeline:
    model_type: str
    stages: list[StageConfig]
    async_chunk: bool = False
    connectors: dict | None = None
    edges: list[dict] | None = None
```

提供拓扑验证方法：

```python
def validate_pipeline(self) -> list[str]:
    # 检查：
    # 1. 所有阶段 ID 唯一
    # 2. 所有 input_sources 引用有效阶段
    # 3. 至少一个入口点（无输入源的阶段）
```

### StageConfigFactory — 配置工厂

```python
class StageConfigFactory:
    PIPELINE_MODELS = {
        "qwen3_omni_moe": "qwen3_omni",
        "qwen2_5_omni": "qwen2_5_omni",
        "bagel": "bagel",
        "qwen3_tts": "qwen3_tts",
        "mimo_audio": "mimo_audio",
        "glm-image": "glm_image",
        "cosyvoice3": "cosyvoice3",
        "mammothmoda2": "mammoth_moda2",
    }
```

核心方法 `create_from_model()` 的工作流程：

```
model路径 → _auto_detect_model_type() → 查找 PIPELINE_MODELS
    → get_pipeline_path() → _parse_pipeline_yaml()
    → _merge_cli_overrides() → list[StageConfig]
```

### CLI 覆盖合并

```python
@classmethod
def _merge_cli_overrides(cls, stage, cli_overrides):
    result = {}
    # 全局覆盖（排除内部键和阶段特定键）
    for key, value in cli_overrides.items():
        if key in cls._INTERNAL_KEYS:
            continue
        if re.match(r"stage_\d+_", key):
            continue
        if value is not None:
            result[key] = value
    # 阶段特定覆盖 (--stage-N-*)
    stage_prefix = f"stage_{stage.stage_id}_"
    for key, value in cli_overrides.items():
        if key.startswith(stage_prefix) and value is not None:
            param_name = key[len(stage_prefix):]
            result[param_name] = value
    return result
```

支持两种覆盖方式：
- 全局覆盖：适用于所有阶段
- 阶段特定覆盖：通过 `--stage-N-参数名` 格式指定

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `StageType` | 枚举 | 阶段类型（LLM / DIFFUSION） |
| `StageConfig` | 数据类 | 单阶段的完整配置 |
| `ModelPipeline` | 数据类 | 完整流水线定义 |
| `StageConfigFactory` | 工厂类 | 加载 YAML + 合并 CLI 覆盖 |
| `get_pipeline_path` | 函数 | 获取流水线 YAML 路径 |
| `create_from_model` | 类方法 | 从模型路径创建阶段配置 |
| `create_default_diffusion` | 类方法 | 创建默认扩散阶段 |
| `_auto_detect_model_type` | 类方法 | 自动检测模型类型 |
| `_parse_pipeline_yaml` | 类方法 | 解析 YAML 文件 |

## 与其他模块的关系

- 使用 `yaml_util.py` 加载和处理 YAML 配置
- 流水线 YAML 文件位于 `model_executor/models/<model>/pipeline.yaml`
- 被引擎层用于初始化多阶段模型部署
- `StageConfig.to_omegaconf()` 输出被 `OmniModelConfig` 消费

## 总结

`stage_config.py` 是 vllm-omni 流水线架构的配置基石。通过 YAML + CLI 的分层配置设计，实现了"模型开发者定义拓扑，用户调整运行参数"的清晰分工。支持的模型包括 Qwen-Omni、Bagel、GLM-Image、CosyVoice3 等多种多模态架构。
