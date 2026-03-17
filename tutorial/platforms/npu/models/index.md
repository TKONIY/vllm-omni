# npu/models/ 子模块索引

## 模块概述

`npu/models/` 存放 NPU 平台专用的模型算子实现。这些算子是通用（CUDA）实现的 Ascend NPU 等效替代，在功能相同的前提下利用 NPU 特有的硬件加速能力。

## 文件列表

| 文件 | 说明 |
|------|------|
| `__init__.py` | 空初始化文件 |
| [hunyuan_fused_moe.py.md](./hunyuan_fused_moe.py.md) | HunyuanFusedMoE 的 Ascend NPU 实现 |

## 设计原理

平台专用算子通过 `OmniPlatform.get_diffusion_model_impl_qualname()` 方法进行注册和分发。当模型需要某个算子（如 `hunyuan_fused_moe`）时，平台层根据当前硬件返回对应的实现类全限定名，由上层代码动态加载。
