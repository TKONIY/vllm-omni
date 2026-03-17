# `__init__.py` — LTX2 模块入口

## 文件概述

该文件是 `ltx2` 子包的初始化模块，导出 LTX2 视频生成模型的核心组件，包括文本到视频管线、图像到视频管线、Transformer 模型以及配置加载工具。

## 关键代码解析

```python
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import (
    LTX2Pipeline, create_transformer_from_config, get_ltx2_post_process_func, load_transformer_config,
)
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_image2video import LTX2ImageToVideoPipeline
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `LTX2Pipeline` | 类 | 文本到视频生成管线 |
| `LTX2ImageToVideoPipeline` | 类 | 图像到视频生成管线 |
| `LTX2VideoTransformer3DModel` | 类 | 3D 视频 Transformer 模型 |
| `get_ltx2_post_process_func` | 函数 | 获取后处理函数 |
| `load_transformer_config` | 函数 | 加载 Transformer 配置 |
| `create_transformer_from_config` | 函数 | 从配置创建 Transformer |

## 与其他模块的关系

- 从 `ltx2_transformer`、`pipeline_ltx2`、`pipeline_ltx2_image2video` 导入
- 被外部模型注册系统引用

## 总结

该文件导出了 LTX2 视频生成的所有核心组件，支持文本到视频和图像到视频两种生成模式。
