# `pipeline_longcat_image.py` — LongCat 文本到图像生成管线

## 文件概述

该文件实现了 LongCat Image 的文本到图像生成管线（Pipeline），整合了文本编码器（Qwen2.5-VL）、VAE、调度器和 Transformer 模型。管线支持 prompt rewrite（提示词改写）、Classifier-Free Guidance（CFG）、CFG 归一化等高级功能。

## 关键代码解析

### Prompt Rewrite 功能

```python
def rewire_prompt(self, prompt, device):
    # 根据语言选择系统提示
    if language == "zh":
        question = SYSTEM_PROMPT_ZH + f"\n用户输入为：{each_prompt}\n改写后的prompt为："
    else:
        question = SYSTEM_PROMPT_EN + f"\nUser Input: {each_prompt}\nRewritten prompt:"
    # 使用 Qwen2.5-VL 生成改写后的提示词
    generated_ids = self.text_encoder.generate(**inputs, max_new_tokens=self.tokenizer_max_length)
```

利用文本编码器模型自动改写用户输入的提示词，提升生成质量。

### 引号感知的文本编码

```python
def split_quotation(prompt, quote_pairs=None):
    # 识别引号包裹的内容，按字符逐个编码
    for clean_prompt_sub, matched in split_quotation(each_prompt):
        if matched:
            for sub_word in clean_prompt_sub:
                tokens = self.tokenizer(sub_word, add_special_tokens=False)["input_ids"]
```

对引号内的文本按字符逐个编码，确保文字渲染任务中的精确控制。

### CFG 归一化

```python
def cfg_normalize_function(self, noise_pred, comb_pred, cfg_renorm_min=0.0):
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
    noise_pred = comb_pred * scale
```

对 CFG 结合后的噪声预测进行范数归一化，防止过度偏离条件预测的分布。

### 去噪循环

```python
for i, t in enumerate(timesteps):
    noise_pred = self.predict_noise_maybe_with_cfg(
        do_true_cfg=self.do_classifier_free_guidance,
        true_cfg_scale=guidance_scale,
        positive_kwargs=positive_kwargs,
        negative_kwargs=negative_kwargs,
        cfg_normalize=enable_cfg_renorm,
    )
    latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, ...)
```

使用 `CFGParallelMixin` 提供的方法自动处理 CFG 并行。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `LongCatImagePipeline` | 类 | 文本到图像生成管线 |
| `get_longcat_image_post_process_func` | 函数 | 返回 VAE 后处理函数 |
| `calculate_shift` | 函数 | 根据序列长度计算 timestep shift |
| `split_quotation` | 函数 | 引号感知的字符串分割 |
| `prepare_pos_ids` | 函数 | 准备位置 ID（文本/图像） |
| `retrieve_timesteps` | 函数 | 获取调度器的时间步 |
| `get_prompt_language` | 函数 | 检测提示词语言 |

## 与其他模块的关系

- 引用 `LongCatImageTransformer2DModel` 作为核心去噪模型
- 继承 `CFGParallelMixin` 获得 CFG 并行能力
- 使用 `DiffusersPipelineLoader` 加载权重
- 使用 Qwen2.5-VL 作为文本编码器
- 使用 diffusers 的 `AutoencoderKL` 作为 VAE
- 使用 `FlowMatchEulerDiscreteScheduler` 作为调度器

## 总结

该管线实现了完整的文本到图像生成流程，特色功能包括：自动提示词改写、引号感知编码（适合文字渲染）、CFG 归一化、以及通过 `CFGParallelMixin` 支持的 CFG 并行推理。管线与 vLLM-Omni 的请求系统深度集成，从 `OmniDiffusionRequest` 中提取所有参数。
