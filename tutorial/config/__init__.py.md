# `config/__init__.py` — 配置模块入口

## 文件概述

`config/__init__.py` 是配置模块的入口文件，统一导出所有配置相关的类和函数。

## 关键代码解析

```python
from vllm_omni.config.lora import LoRAConfig
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.config.stage_config import (
    ModelPipeline, StageConfig, StageConfigFactory, StageType,
)
from vllm_omni.config.yaml_util import (
    create_config, load_yaml_config, merge_configs, to_dict,
)
```

纯导入文件，提供统一的导入入口。

## 导出列表

| 名称 | 来源 | 用途 |
|------|------|------|
| `OmniModelConfig` | `model.py` | 多阶段模型配置 |
| `LoRAConfig` | `lora.py` | LoRA 配置 |
| `StageConfig` | `stage_config.py` | 单阶段配置 |
| `StageConfigFactory` | `stage_config.py` | 阶段配置工厂 |
| `ModelPipeline` | `stage_config.py` | 流水线定义 |
| `StageType` | `stage_config.py` | 阶段类型枚举 |
| `create_config` | `yaml_util.py` | 创建 OmegaConf 配置 |
| `load_yaml_config` | `yaml_util.py` | 加载 YAML 文件 |
| `merge_configs` | `yaml_util.py` | 合并多个配置 |
| `to_dict` | `yaml_util.py` | 配置转字典 |

## 总结

标准的 Python 包入口，将分散在子模块中的配置组件统一到 `vllm_omni.config` 命名空间下。
