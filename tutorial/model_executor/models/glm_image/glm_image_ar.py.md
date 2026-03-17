# `glm_image_ar.py` — GLM-Image 自回归模型

## 文件概述

GLM-Image 的完整 AR 模型实现，包含多模态处理器、视觉编码器、VQ-VAE、Qwen2 LLM backbone 以及 M-RoPE 位置编码。文件约 900 行，是 GLM-Image 模块中唯一的核心文件。

## 关键代码解析

### 1. 多模态处理信息

```python
class GlmImageProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs):
        """自动解析处理器路径：vision_language_encoder/ → processor/"""
        if model_path.endswith("vision_language_encoder"):
            base_path = os.path.dirname(model_path)
            processor_path = os.path.join(base_path, "processor")
        return GlmImageProcessor.from_pretrained(processor_path)

    def get_supported_mm_limits(self):
        return {"image": 1, "img2img": 1}  # 支持 image 和 img2img 别名
```

### 2. HF 处理器调用

```python
class GlmImageMultiModalProcessor(BaseMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        if not mm_data.get("images"):
            # t2i 模式：使用 apply_chat_template 构建 prompt
            hf_inputs = processor.apply_chat_template(
                messages, target_h=target_h, target_w=target_w)
        else:
            # i2i 模式：将图像嵌入 content，分离 source/target grid
            hf_inputs["mrope_image_grid_thw"] = image_grid_thw  # 完整 grid（M-RoPE 用）
            hf_inputs["image_grid_thw"] = source_grids           # 源 grid（MM 编码用）
```

### 3. Prompt 替换逻辑

```python
def _get_prompt_updates(self, mm_items, ...):
    """i2i: 将 <|image|> 替换为 grid_h * grid_w 个 image_token"""
    def get_image_replacement(item_idx):
        grid_thw = out_mm_kwargs["image"][item_idx]["image_grid_thw"]
        num_tokens = grid_thw[1] * grid_thw[2]  # height * width patches
        return [image_token_id] * num_tokens
```

### 4. 模型核心功能

由于文件很长，模型类的核心功能包括：

- **视觉编码器**：处理 i2i 源图像，输出 patch 级别特征
- **VQ-VAE**：将视觉特征量化为离散 token
- **Qwen2 LLM**：自回归生成图像 token 序列
- **M-RoPE**：基于 `image_grid_thw` 计算多维位置编码
- **OmniOutput**：输出包装为 `OmniOutput` 格式

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `GlmImagePixelInputs` | Schema | 像素输入的 TensorSchema |
| `GlmImageDataParser` | 类 | 数据解析器，img2img 映射为 image |
| `GlmImageProcessingInfo` | 类 | 处理信息（路径解析、配置） |
| `GlmImageDummyInputsBuilder` | 类 | 虚拟输入构建器 |
| `GlmImageMultiModalProcessor` | 类 | 多模态处理器核心 |
| `GlmImageForConditionalGeneration` | 类 | 完整的 AR 生成模型 |

## 与其他模块的关系

- 使用 vLLM 的 `Qwen2MLP`、`QKVParallelLinear` 等标准组件
- 实现 `SupportsMRoPE`、`SupportsMultiModal`、`SupportsPP` 接口
- 输出 `OmniOutput` 供 vLLM-omni 的 AR 调度器消费
- 使用 HuggingFace `transformers.models.glm_image` 的配置和处理器类

## 总结

GLM-Image 的 AR 模型是一个功能完整的图像生成系统，核心创新在于使用 M-RoPE 处理图像的空间位置关系，以及精心设计的 i2i 处理流程（分离 source/target grid、自动路径解析等）。
