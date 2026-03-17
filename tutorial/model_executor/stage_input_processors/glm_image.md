# `glm_image.py` -- GLM-Image AR 到扩散阶段处理器

## 文件概述

`glm_image.py` 实现了 GLM-Image 模型从 AR 阶段到 Diffusion 阶段的数据转换。核心工作包括：从 AR 模型生成的 token IDs 中解析先验图像 token、将 32x 下采样的 token 上采样到 16x，以及组装扩散阶段所需的完整输入。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/glm_image.py`

## 关键代码解析

### 图像 Token 上采样

GLM-Image 的 AR 模型生成 32x 下采样的 token，但 DiT 期望 16x 下采样，因此需要 2x 最近邻上采样：

```python
def _upsample_token_ids(token_ids, token_h, token_w):
    token_ids = token_ids.view(1, 1, token_h, token_w)
    token_ids = torch.nn.functional.interpolate(
        token_ids.float(), scale_factor=2, mode="nearest"
    ).to(dtype=torch.long)
    return token_ids.view(-1)
```

### Token 解析逻辑

AR 模型在 text2img 模式下生成两部分 token：

```
[小预览图 token (16x16=256)] + [大图 token (32x32=1024)] + [EOS (16385)]
```

`_parse_generated_tokens` 函数处理多种情况：

1. **text2img**: 跳过前 `small_image_tokens` 个 token，提取大图 token
2. **img2img**: 可能只有大图 token，也可能包含小图 + 大图
3. **token 不足**: 逐级尝试更小的网格，最终使用平方根近似

```python
if actual_tokens >= small_image_tokens + large_image_tokens:
    # 标准 text2img：提取大图 token
    prior_token_ids_d32 = token_tensor[small_image_tokens:small_image_tokens + large_image_tokens]
elif actual_tokens >= large_image_tokens:
    # img2img：直接使用前面的 token
    prior_token_ids_d32 = token_tensor[:large_image_tokens]
```

### ar2diffusion -- 主处理函数

```python
def ar2diffusion(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False):
```

完整流程：
1. 从 AR 阶段输出中提取生成的 token IDs
2. 从原始 prompt 中获取目标尺寸（height/width）
3. 调用 `_parse_generated_tokens` 解析并上采样 token
4. 提取 `prior_token_image_ids`（img2img 模式下的输入图像 VQ-VAE token）
5. 组装扩散输入字典（prior_token_ids + 尺寸 + 引导参数）
6. 如果需要，传递 PIL 图像给扩散模型

输出格式：
```python
diffusion_input = {
    "prompt": text_prompt,
    "height": pixel_h,
    "width": pixel_w,
    "extra": {
        "prior_token_ids": prior_token_ids,          # 上采样后的先验 token
        "prior_token_image_ids": prior_token_image_ids,  # img2img 的参考图 token
    },
    "seed": ..., "num_inference_steps": ..., "guidance_scale": ...
}
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ar2diffusion` | 函数 | AR 输出 -> Diffusion 输入转换（同步） |
| `_parse_generated_tokens` | 函数 | 解析 AR 生成的 token，提取先验 token |
| `_upsample_token_ids` | 函数 | 2x 最近邻上采样（32x -> 16x） |

## 与其他模块的关系

- **stage_configs/glm_image.yaml**: `custom_process_input_func` 引用 `glm_image.ar2diffusion`
- **models/glm_image/**: AR 模型输出 token IDs 和 `prior_token_image_ids`
- **engine/**: 扩散阶段引擎接收此处理器输出的字典作为输入

## 总结

`glm_image.py` 是 GLM-Image 图像生成流水线中 AR->Diffusion 的关键桥梁。其核心复杂度在于 Token 解析：需要处理 text2img/img2img 两种模式、小图/大图的双层 token 结构、以及 32x->16x 的上采样转换。该处理器还包含了对异常情况（token 数量不足）的多级降级处理策略。
