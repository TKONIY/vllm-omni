# `mammoth_moda2.py` -- MammothModa2 AR 到 DiT 处理器

## 文件概述

`mammoth_moda2.py` 实现了 MammothModa2 多模态生成模型从 AR 阶段到 DiT（Diffusion Transformer）阶段的数据转换。核心工作是从 AR 阶段的隐藏状态中提取文本条件和图像条件嵌入，并组装为 DiT 所需的输入格式。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/mammoth_moda2.py`

## 关键代码解析

### ar2dit 函数

```python
def ar2dit(stage_list, engine_input_source, prompts=None, requires_multimodal_data=False):
    """Convert AR stage outputs to DiT stage inputs."""
```

处理流程：

**1. 提取 AR 输出的隐藏状态**

```python
full_hidden_states = mm_output["latent"]
full_token_ids = prompt_token_ids + gen_token_ids
```

注意：`gen_token_ids` 排除了最后一个 token（因为它没有对应的隐藏状态）。

**2. 构建条件掩码**

通过 token 类型区分文本条件和图像条件：

```python
# 问题区域（prompt）中的非视觉、非生成 token -> 文本条件
text_condition_token_mask = questions_mask & ~(visual_token_mask | gen_token_mask) & attention_mask

# 答案区域中的生成 token -> 图像条件
image_condition_token_mask = answers_mask & gen_token_mask & attention_mask
```

其中：
- `questions_mask`: prompt 部分的 token
- `answers_mask`: 生成部分的 token
- `gen_token_mask`: token ID >= `gen_vocab_start_index` 的 token
- `visual_token_mask`: 特殊视觉 token（`<|image_pad|>`, `<|video_pad|>` 等）

**3. 提取条件嵌入**

```python
text_condition = full_hidden_states[text_condition_token_mask]
image_condition = full_hidden_states[image_condition_token_mask]

text_prompt_embeds = text_condition.to(dtype=torch.float32).contiguous()
image_prompt_embeds = image_condition.to(dtype=torch.float32).contiguous()
```

**4. 组装 DiT 输入**

```python
additional_information = {
    "text_prompt_embeds": text_prompt_embeds,
    "image_prompt_embeds": image_prompt_embeds,
    "image_height": [int(image_height)],
    "image_width": [int(image_width)],
    "text_guidance_scale": [float(text_guidance_scale)],
    "cfg_range": [float(cfg_range[0]), float(cfg_range[1])],
    "num_inference_steps": [int(num_inference_steps)],
}
```

DiT 阶段的 `prompt_token_ids` 被设为 `[0]`（占位符），实际输入通过 `additional_information` 传递。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ar2dit` | 函数 | AR 输出 -> DiT 输入转换（同步） |

## 与其他模块的关系

- **stage_configs/mammoth_moda2.yaml**: `custom_process_input_func` 引用 `mammoth_moda2.ar2dit`
- **models/mammoth_moda2/**: AR 模型输出包含 `latent` 隐藏状态；DiT 模型消费 `text_prompt_embeds` 和 `image_prompt_embeds`
- **inputs/data.py**: 使用 `OmniTokensPrompt` 封装 DiT 输入

## 总结

`mammoth_moda2.py` 通过精确的 token 类型掩码（文本/视觉/生成 token 的区分）从 AR 模型的隐藏状态中分离出文本条件和图像条件嵌入，为 DiT 扩散模型提供了双路条件引导。这种基于隐藏状态而非 token ID 的条件传递方式，是 MammothModa2 与 GLM-Image（基于 token ID 传递）的核心差异。
