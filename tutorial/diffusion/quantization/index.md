# 量化子模块索引

## 概述

`quantization/` 子模块为扩散模型提供统一的量化支持。它封装了 vLLM 的量化基础设施，目前支持 FP8 和 GGUF 两种量化方法。通过注册表 + 工厂模式，新增量化方法只需添加配置类并注册。

## 架构设计

```
quantization/
├── __init__.py    # 入口：工厂函数、注册表、配置提取
├── base.py        # 抽象基类 DiffusionQuantizationConfig
├── fp8.py         # FP8 量化配置（动态缩放、在线量化）
└── gguf.py        # GGUF 量化配置（反量化 GEMM、分片处理）
```

## 文件列表

| 文件 | 说明 | 文档链接 |
|------|------|----------|
| `__init__.py` | 工厂函数 `get_diffusion_quant_config`、注册表、vLLM 配置提取 | [__init__.md](./__init__.md) |
| `base.py` | 抽象基类，封装 vLLM `QuantizationConfig` | [base.md](./base.md) |
| `fp8.py` | FP8 量化：动态激活缩放、BF16 在线量化、块级权重量化 | [fp8.md](./fp8.md) |
| `gguf.py` | GGUF 量化：反量化 GEMM、QKV 分片反量化、自定义线性方法 | [gguf.md](./gguf.md) |

## 量化方法对比

| 方法 | 精度 | 权重来源 | 需要校准 | GPU 要求 |
|------|------|----------|----------|----------|
| FP8 | W8A8 / 仅权重 | BF16 检查点 + 在线量化 | 否（动态缩放） | SM 75+ |
| GGUF | 多种（Q4/Q5/Q8等） | 预量化 GGUF 文件 | 否 | 通用 |

## 使用示例

```python
from vllm_omni.diffusion.quantization import get_diffusion_quant_config, get_vllm_quant_config_for_layers

# 创建 FP8 配置
config = get_diffusion_quant_config("fp8")

# 获取 vLLM 配置用于线性层
vllm_config = get_vllm_quant_config_for_layers(config)
```
