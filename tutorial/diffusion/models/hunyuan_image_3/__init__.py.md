# `__init__.py` — HunyuanImage3 模块初始化与导出

## 文件概述

该文件是 HunyuanImage3 图像生成模型子包的入口文件，统一导出核心组件。HunyuanImage3 是一个基于自回归（AR）架构的统一多模态模型，将图像生成建模为 token 序列预测，结合了 MoE（混合专家）和扩散去噪。

## 关键代码解析

```python
from vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe import HunyuanFusedMoE
from vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_image_3_transformer import (
    HunyuanImage3Model,
    HunyuanImage3Text2ImagePipeline,
)
from vllm_omni.diffusion.models.hunyuan_image_3.pipeline_hunyuan_image_3 import (
    HunyuanImage3Pipeline,
)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `HunyuanImage3Pipeline` | 类 | 完整推理管线（继承 GenerationMixin） |
| `HunyuanImage3Model` | 类 | Decoder-only Transformer 模型 |
| `HunyuanImage3Text2ImagePipeline` | 类 | 文本到图像扩散管线 |
| `HunyuanFusedMoE` | 类 | 融合 MoE 层（平台自适应） |

## 与其他模块的关系

- 作为 `vllm_omni.diffusion.models.hunyuan_image_3` 包入口
- `HunyuanImage3Pipeline` 被上层模型加载器调用

## 总结

`__init__.py` 集中导出了 HunyuanImage3 的四个核心组件：推理管线、Transformer 模型、扩散管线和 MoE 层。
