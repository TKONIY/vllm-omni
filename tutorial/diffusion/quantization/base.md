# `base.py` — 量化配置基类

## 文件概述

`base.py` 定义了 `DiffusionQuantizationConfig` 抽象基类，作为扩散模型所有量化配置的统一接口。它是 vLLM 量化配置的薄封装层，允许扩散模型定义自己的默认值和扩展点。

## 关键代码解析

### 类定义与设计

```python
class DiffusionQuantizationConfig(ABC):
    """Base class for diffusion model quantization configurations."""

    # 子类应设置对应的 vLLM QuantizationConfig 类
    quant_config_cls: ClassVar[type["QuantizationConfig"] | None] = None

    # 底层 vLLM 配置实例
    _vllm_config: "QuantizationConfig | None" = None

    def get_name(self) -> str:
        if self._vllm_config is not None:
            return self._vllm_config.get_name()
        raise NotImplementedError

    def get_vllm_quant_config(self) -> "QuantizationConfig | None":
        return self._vllm_config

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        if cls.quant_config_cls is not None:
            return cls.quant_config_cls.get_min_capability()
        return 80  # Ampere 默认值
```

设计要点：
- **`quant_config_cls`**：类变量，子类应设置为对应的 vLLM 量化配置类（如 `Fp8Config`）。
- **`_vllm_config`**：实例变量，持有实际的 vLLM 配置实例。
- **委托模式**：`get_name()` 和 `get_min_capability()` 默认委托给 vLLM 配置。

### 子类实现规范

子类需要：
1. 设置 `quant_config_cls` 为 vLLM 的量化配置类。
2. 在 `__init__` 中创建 `self._vllm_config` 实例。
3. 可选地覆盖 `get_name()` 和 `get_min_capability()`。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionQuantizationConfig` | 抽象类 | 量化配置基类 |
| `get_name` | 方法 | 返回量化方法名称（如 "fp8"） |
| `get_vllm_quant_config` | 方法 | 返回底层 vLLM 量化配置实例 |
| `get_supported_act_dtypes` | 类方法 | 返回支持的激活数据类型 |
| `get_min_capability` | 类方法 | 返回最低 GPU 计算能力要求 |

## 与其他模块的关系

- **`fp8.py`**：`DiffusionFp8Config` 继承此基类。
- **`gguf.py`**：`DiffusionGgufConfig` 继承此基类。
- **`__init__.py`**：注册表中使用此类作为类型约束。
- **vLLM 量化层**：通过 `get_vllm_quant_config()` 获取配置传给线性层。

## 总结

`DiffusionQuantizationConfig` 通过组合模式封装了 vLLM 的量化配置，提供了扩散模型特定的接口层。这种设计使得新增量化方法只需继承此基类并配置 vLLM 参数，无需修改框架代码。
