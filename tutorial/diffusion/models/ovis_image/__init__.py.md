# `__init__.py` — Ovis Image 模块入口

## 文件概述

该文件是 `ovis_image` 子包的初始化模块，导出 Ovis Image 7B 模型的核心组件。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OvisImagePipeline` | 类 | Ovis Image 生成管线 |
| `OvisImageTransformer2DModel` | 类 | Ovis Image Transformer 模型 |
| `get_ovis_image_post_process_func` | 函数 | 获取后处理函数 |

## 与其他模块的关系

- 从 `ovis_image_transformer` 和 `pipeline_ovis_image` 导入

## 总结

该文件导出了 Ovis Image 模型的三个核心组件，提供清晰的公共 API。
