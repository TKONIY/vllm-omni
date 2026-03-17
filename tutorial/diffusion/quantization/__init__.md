# `__init__.py` — 量化模块入口与工厂函数

## 文件概述

`quantization/__init__.py` 是扩散模型量化子模块的入口文件。它提供了一个统一的工厂接口，用于根据量化方法名称创建对应的量化配置对象。该模块封装了 vLLM 的量化基础设施，同时允许扩散模型使用特定的默认值和优化策略。

## 关键代码解析

### 1. 量化方法注册表

```python
_QUANT_CONFIG_REGISTRY: dict[str, type[DiffusionQuantizationConfig]] = {
    "fp8": DiffusionFp8Config,
    "gguf": DiffusionGgufConfig,
}

SUPPORTED_QUANTIZATION_METHODS = list(_QUANT_CONFIG_REGISTRY.keys())
```

注册表使用字典将量化方法名映射到对应的配置类。新增量化方法只需创建配置类并在此注册即可。

### 2. 量化配置工厂函数

```python
def get_diffusion_quant_config(quantization: str | None, **kwargs) -> DiffusionQuantizationConfig | None:
    if quantization is None or quantization.lower() == "none":
        return None
    quantization = quantization.lower()
    if quantization not in _QUANT_CONFIG_REGISTRY:
        raise ValueError(f"Unknown quantization method: {quantization!r}.")
    config_cls = _QUANT_CONFIG_REGISTRY[quantization]
    return config_cls(**kwargs)
```

根据字符串参数创建量化配置，传入 `None` 或 `"none"` 则禁用量化。额外的关键字参数会传递给具体的配置类构造函数。

### 3. vLLM 配置提取

```python
def get_vllm_quant_config_for_layers(diffusion_quant_config):
    if diffusion_quant_config is None:
        return None
    return diffusion_quant_config.get_vllm_quant_config()
```

从扩散模型的量化配置中提取底层的 vLLM `QuantizationConfig`，用于传递给 vLLM 的线性层（如 `QKVParallelLinear`）。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_diffusion_quant_config` | 函数 | 量化配置工厂函数，根据方法名创建配置对象 |
| `get_vllm_quant_config_for_layers` | 函数 | 提取 vLLM 量化配置用于线性层初始化 |
| `_QUANT_CONFIG_REGISTRY` | 字典 | 量化方法注册表 |
| `SUPPORTED_QUANTIZATION_METHODS` | 列表 | 支持的量化方法名称列表 |

## 与其他模块的关系

- **`base.py`**：导入 `DiffusionQuantizationConfig` 基类。
- **`fp8.py`**：导入 `DiffusionFp8Config` 配置类。
- **`gguf.py`**：导入 `DiffusionGgufConfig` 配置类。
- **扩散 pipeline**：pipeline 初始化时调用工厂函数获取量化配置。
- **模型定义**：模型层通过 `get_vllm_quant_config_for_layers` 获取 vLLM 配置。

## 总结

此模块通过注册表 + 工厂函数模式，提供了量化方法的可扩展入口。使用者只需指定量化方法名即可获得正确的配置对象，无需关心具体实现细节。
