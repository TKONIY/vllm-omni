# config 模块教程 — 配置系统

## 模块概述

`config/` 模块是 vllm-omni 的配置中心，负责定义模型配置、流水线阶段配置和 YAML 配置加载。它将模型开发者定义的流水线拓扑（YAML 文件）与用户的运行时参数（CLI）统一管理。

## 架构图

```
config/
├── __init__.py           # 模块入口，统一导出
├── lora.py               # LoRA 配置（转发 vLLM 实现）
├── model.py              # OmniModelConfig — 模型配置核心
├── stage_config.py        # 流水线阶段配置系统
└── yaml_util.py           # OmegaConf YAML 工具封装

                  ┌──────────────┐
                  │  pipeline.yaml│  ← 模型开发者定义
                  └──────┬───────┘
                         │ load_yaml_config()
                         ▼
                  ┌──────────────┐
                  │ StageConfig   │  × N 个阶段
                  │ Factory       │
                  └──────┬───────┘
                         │ _merge_cli_overrides()
                         ▼
              ┌──────────────────────┐
              │  ModelPipeline        │
              │  ├─ StageConfig[0]    │  (thinker)
              │  ├─ StageConfig[1]    │  (talker)
              │  └─ ...               │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  OmniModelConfig      │  ← 每个阶段一个
              │  (extends ModelConfig) │
              └──────────────────────┘
```

## 模块文档索引

| 文件 | 说明 |
|------|------|
| [__init__.py.md](./__init__.py.md) | 模块入口与导出 |
| [lora.py.md](./lora.py.md) | LoRA 配置 |
| [model.py.md](./model.py.md) | OmniModelConfig 模型配置 |
| [stage_config.py.md](./stage_config.py.md) | 流水线阶段配置系统 |
| [yaml_util.py.md](./yaml_util.py.md) | YAML 工具封装 |
