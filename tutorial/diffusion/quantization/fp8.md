# `fp8.py` — FP8 量化配置

## 文件概述

`fp8.py` 实现了 `DiffusionFp8Config`，为扩散 transformer 模型提供 FP8 量化支持。它使用动态激活缩放（无需校准数据集），支持从 BF16/FP16 检查点进行在线权重量化。

## 关键代码解析

```python
class DiffusionFp8Config(DiffusionQuantizationConfig):
    """FP8 quantization config optimized for diffusion transformers."""

    quant_config_cls = Fp8Config

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        weight_block_size: list[int] | None = None,
        ignored_layers: list[str] | None = None,
    ):
        self.activation_scheme = activation_scheme
        self.weight_block_size = weight_block_size
        self.ignored_layers = ignored_layers or []

        self._vllm_config = Fp8Config(
            is_checkpoint_fp8_serialized=False,  # 从 BF16 在线量化
            activation_scheme=activation_scheme,
            weight_block_size=weight_block_size,
            ignored_layers=ignored_layers,
        )
```

关键设计决策：
- **`is_checkpoint_fp8_serialized=False`**：权重文件为 BF16/FP16 格式，在加载时动态量化为 FP8，而非使用预量化的检查点。
- **`activation_scheme="dynamic"`**：默认使用动态逐 token 缩放，避免了静态校准的复杂性。
- **设备兼容性**：Turing (SM 75+) 使用 Marlin 内核进行仅权重 FP8，Ada/Hopper (SM 89+) 支持完整的 W8A8 FP8 硬件加速。内核选择由 vLLM 自动完成。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionFp8Config` | 类 | FP8 量化配置，支持动态/静态激活缩放和块级权重量化 |
| `activation_scheme` | 参数 | 激活量化方案："dynamic"（默认）或 "static" |
| `weight_block_size` | 参数 | 块级权重量化的块大小，格式 `[block_n, block_k]` |
| `ignored_layers` | 参数 | 跳过量化的层名称模式列表 |

## 与其他模块的关系

- **`base.py`**：继承 `DiffusionQuantizationConfig` 基类。
- **vLLM `Fp8Config`**：封装并委托 vLLM 的 FP8 配置。
- **`__init__.py`**：在量化方法注册表中注册为 `"fp8"`。
- **`model_loader/`**：加载器在加载权重后调用 `process_weights_after_loading` 触发 FP8 量化。

## 总结

`DiffusionFp8Config` 以最小配置成本为扩散模型提供了 FP8 量化能力。默认的动态缩放 + 在线量化模式使用户无需准备校准数据或预量化检查点，只需将 BF16 模型直接以 FP8 精度运行即可获得显存和计算收益。
