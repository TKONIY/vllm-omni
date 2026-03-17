# `pipeline_flux2.py` -- Flux 2 推理管线

## 文件概述

实现 `Flux2Pipeline`，Flux 2 的完整推理管线。该管线使用 Mistral3 多模态模型作为文本/图像编码器，支持文本到图像、图像到图像（超分辨率）等任务。实现了 `SupportImageInput` 接口以声明图像输入能力。

**文件路径**: `vllm_omni/diffusion/models/flux2/pipeline_flux2.py`

## 关键代码解析

### 管线特色

- **Mistral3 编码器**: 使用 `Mistral3ForConditionalGeneration` 作为文本/图像条件编码器
- **Flux2 专用 VAE**: 使用 `AutoencoderKLFlux2` 解码潜变量
- **图像超分辨率**: 支持 upsampling 模式，对输入图像进行超分辨率处理
- **System Message**: 不同模式（t2i、upsampling_t2i、upsampling_i2i）使用不同的系统提示词

### Flux2ImageProcessor

```python
class Flux2ImageProcessor(VaeImageProcessor):
    # 自定义图像处理器，支持 Flux2 的特定预处理需求
```

### encode_prompt 方法

使用 Mistral3 的 PixtralProcessor 处理文本和图像输入，生成条件嵌入。

### CFG 并行支持

继承 `CFGParallelMixin`，支持多 GPU 分离计算正向/负向条件预测。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Flux2Pipeline` | nn.Module | Flux 2 完整推理管线 |
| `Flux2ImageProcessor` | VaeImageProcessor | 图像预处理器 |
| `get_flux2_post_process_func` | 工厂函数 | 图像后处理函数 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 依赖 | `flux2_transformer.py` | Flux 2 Transformer 模型 |
| 实现 | `interface.SupportImageInput` | 声明图像输入能力 |
| 混入 | `CFGParallelMixin` | CFG 并行支持 |
| 依赖 | transformers | Mistral3 编码器 |

## 总结

`Flux2Pipeline` 是 Flux 2 的高级推理管线，最大特色是使用多模态 Mistral3 模型作为条件编码器，天然支持图像和文本混合输入。管线支持文本到图像和图像超分辨率两种模式，通过不同的系统提示词和处理流程切换。
