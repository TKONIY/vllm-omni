# `__init__.py` — Helios 模块初始化与导出

## 文件概述

该文件是 Helios 视频生成模型子包的入口文件，负责从子模块中导入核心组件并通过 `__all__` 统一对外暴露。Helios 是一个基于 Wan2.2 架构扩展的分块长视频生成模型。

## 关键代码解析

```python
from .helios_transformer import HeliosTransformer3DModel
from .pipeline_helios import (
    HeliosPipeline,
    create_transformer_from_config,
    get_helios_post_process_func,
    get_helios_pre_process_func,
    load_transformer_config,
)
from .scheduling_helios import HeliosScheduler
```

文件将三个子模块中的关键组件全部集中导出：
- **Transformer 模型**：`HeliosTransformer3DModel`
- **推理管线**：`HeliosPipeline` 及其辅助函数
- **调度器**：`HeliosScheduler`

## 核心类/函数

| 名称 | 类型 | 来源模块 | 说明 |
|------|------|----------|------|
| `HeliosTransformer3DModel` | 类 | `helios_transformer` | 3D Transformer 核心模型 |
| `HeliosPipeline` | 类 | `pipeline_helios` | 完整推理管线 |
| `HeliosScheduler` | 类 | `scheduling_helios` | 统一调度器（Euler/UniPC/DMD） |
| `load_transformer_config` | 函数 | `pipeline_helios` | 加载 Transformer 配置 |
| `create_transformer_from_config` | 函数 | `pipeline_helios` | 从配置创建 Transformer 实例 |
| `get_helios_pre_process_func` | 函数 | `pipeline_helios` | 获取预处理函数 |
| `get_helios_post_process_func` | 函数 | `pipeline_helios` | 获取后处理函数 |

## 与其他模块的关系

- 作为 `vllm_omni.diffusion.models.helios` 包的入口，被上层模型注册机制调用
- 向外暴露的接口可被 `DiffusersPipelineLoader` 等加载器使用

## 总结

`__init__.py` 作为 Helios 子包的统一入口，将 Transformer 模型、推理管线和调度器三大核心组件集中导出，简化了外部调用路径。
