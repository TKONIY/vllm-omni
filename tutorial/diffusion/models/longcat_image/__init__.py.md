# `__init__.py` — LongCat Image 模块入口

## 文件概述

该文件是 `longcat_image` 子包的初始化模块，负责导出 LongCat Image 模型的核心组件，包括 Transformer 模型、Pipeline 以及后处理函数。

## 关键代码解析

```python
from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import LongCatImageTransformer2DModel
from vllm_omni.diffusion.models.longcat_image.pipeline_longcat_image import (
    LongCatImagePipeline,
    get_longcat_image_post_process_func,
)
```

模块从两个子模块中导出三个核心组件，形成清晰的 API 边界。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `LongCatImageTransformer2DModel` | 类 | LongCat 图像生成的 Transformer 模型 |
| `LongCatImagePipeline` | 类 | 文本到图像的生成管线 |
| `get_longcat_image_post_process_func` | 函数 | 获取图像后处理函数 |

## 与其他模块的关系

- 导入自 `longcat_image_transformer` 和 `pipeline_longcat_image`
- 被外部模型注册系统引用，用于注册 LongCat 图像生成模型

## 总结

该文件是 LongCat Image 模块的统一入口，导出了模型和管线的核心组件，方便外部代码通过包级别的导入访问。
