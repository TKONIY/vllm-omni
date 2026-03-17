# `module_collector.py` — Pipeline 模块发现

## 文件概述

`module_collector.py` 提供了 `ModuleDiscovery` 类，用于从扩散 pipeline 中自动发现 DiT（transformer）、编码器和 VAE 等组件模块。它是卸载系统的前置步骤，为后端提供需要管理的模块列表。

## 关键代码解析

### 1. 模块发现结果

```python
@dataclass
class PipelineModules:
    dits: list[nn.Module]           # DiT/transformer 模块列表
    dit_names: list[str]            # 对应的属性名
    encoders: list[nn.Module]       # 编码器列表
    encoder_names: list[str]        # 对应的属性名
    vae: nn.Module | None = None    # VAE 模块
```

### 2. 模块发现逻辑

```python
class ModuleDiscovery:
    DIT_ATTRS = ["transformer", "transformer_2", "dit", "language_model", "transformer_blocks"]
    ENCODER_ATTRS = ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]
    VAE_ATTRS = ["vae"]

    @staticmethod
    def discover(pipeline: nn.Module) -> PipelineModules:
        # 遍历预定义的属性名列表，查找存在的模块
        for attr in ModuleDiscovery.DIT_ATTRS:
            if hasattr(pipeline, attr):
                module_obj = getattr(pipeline, attr)
                if isinstance(module_obj, nn.Module) and module_obj not in dit_modules:
                    dit_modules.append(module_obj)
```

发现逻辑基于约定的属性名列表：
- **DiT 模块**：按优先级依次检查 `transformer`、`transformer_2`、`dit`、`language_model`、`transformer_blocks`。支持多 DiT 模型（如 MoE 架构）。
- **编码器**：检查 `text_encoder`、`text_encoder_2`、`text_encoder_3`、`image_encoder`。
- **VAE**：检查 `vae` 属性。

去重机制确保同一模块实例不会被重复收集。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `PipelineModules` | 数据类 | 模块发现结果，包含 DiT、编码器和 VAE 列表 |
| `ModuleDiscovery` | 类 | 模块发现工具 |
| `DIT_ATTRS` | 类变量 | DiT 模块的候选属性名 |
| `ENCODER_ATTRS` | 类变量 | 编码器的候选属性名 |
| `VAE_ATTRS` | 类变量 | VAE 的候选属性名 |
| `discover` | 静态方法 | 执行模块发现，返回 `PipelineModules` |

## 与其他模块的关系

- **`layerwise_backend.py`**：调用 `ModuleDiscovery.discover()` 获取 DiT blocks 和编码器。
- **`sequential_backend.py`**：调用 `ModuleDiscovery.discover()` 获取 DiT 和编码器模块。
- **扩散 pipeline**：被发现的目标，需要按照约定的属性命名来注册组件。

## 总结

`ModuleDiscovery` 通过基于属性名约定的发现机制，将 pipeline 的内部结构与卸载逻辑解耦。新增的 pipeline 只需按照约定命名其组件属性，即可自动被卸载系统管理。
