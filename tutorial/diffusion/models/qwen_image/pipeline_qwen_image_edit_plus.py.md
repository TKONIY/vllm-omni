# `pipeline_qwen_image_edit_plus.py` — QwenImage 增强图像编辑管线

## 文件概述

本文件实现了 QwenImage 的增强版图像编辑管线 `QwenImageEditPlusPipeline`。与基础编辑管线相比，增强版在预处理阶段提供了更丰富的图像处理选项，并对编辑过程进行了优化。

## 关键代码解析

### 1. 增强预处理

```python
def get_qwen_image_edit_plus_pre_process_func(od_config):
    def pre_process_func(request):
        # 增强的图像预处理流程
        # 支持更多图像输入格式和尺寸调整策略
        ...
    return pre_process_func
```

### 2. 管线结构

```python
class QwenImageEditPlusPipeline(nn.Module, SupportImageInput, QwenImageCFGParallelMixin):
    def __init__(self, *, od_config, prefix=""):
        # 与基础编辑管线类似，使用相同的组件
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
        self.vae = DistributedAutoencoderKLQwenImage(...)
        self.transformer = QwenImageTransformer2DModel(...)
        self.scheduler = FlowMatchEulerDiscreteScheduler(...)
```

结构上与基础编辑管线一致，主要差异在预处理和后处理逻辑中。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_qwen_image_edit_plus_pre_process_func` | 函数 | 增强编辑预处理工厂 |
| `get_qwen_image_edit_plus_post_process_func` | 函数 | 增强编辑后处理工厂 |
| `QwenImageEditPlusPipeline` | 类 | 增强图像编辑管线 |

## 与其他模块的关系

- 与 `pipeline_qwen_image_edit.py` 共享基础架构
- 继承 `SupportImageInput` 和 `QwenImageCFGParallelMixin`
- 使用相同的 Transformer、VAE 和调度器组件

## 总结

`pipeline_qwen_image_edit_plus.py` 是基础编辑管线的增强版本，提供了更丰富的预处理选项和优化的编辑效果，同时保持与基础管线一致的核心架构。
