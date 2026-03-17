# `pipeline_mammothmoda2_dit.py` — MammothModa2 DiT 管线

## 文件概述

该文件实现了 MammothModa2 的 DiT 生成阶段管线。与其他模型不同，MammothModa2 采用两阶段架构：上游 AR（自回归）阶段生成条件 token，本管线作为下游非自回归阶段接收这些 token 并通过扩散生成图像。管线使用 vLLM 原生的 `VllmConfig` 接口。

## 关键代码解析

### 权重过滤

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "llm_model.": None,  # 忽略 LLM 骨干权重
    }
)
```

只加载 `gen_*` 前缀的权重（DiT/VAE），跳过整个 LLM 骨干。

### 初始化

```python
def __init__(self, *, vllm_config, prefix=""):
    self.gen_vae = AutoencoderKL.from_config(self.config.gen_vae_config)
    self.gen_transformer = Transformer2DModel.from_config(self.config.gen_dit_config)
    # 重建 caption embedder 以匹配 LLM 隐藏维度
    self._reinit_caption_embedder(llm_hidden_size)
    # 可选的 Q-Former 图像条件精化器
    if self.config.gen_image_condition_refiner_config is not None:
        self.gen_image_condition_refiner = SimpleQFormerImageRefiner(...)
```

### 前向传播 — 完整扩散流程

```python
def forward(self, *, inputs_embeds=None, **kwargs):
    # 从 runtime_additional_information 获取条件
    text_cond = info["text_prompt_embeds"]
    image_cond = info["image_prompt_embeds"]

    # 可选：通过 Q-Former 精化图像条件
    if self.gen_image_condition_refiner is not None:
        image_embeds = self.gen_image_condition_refiner(image_embeds, ~image_attention_mask)

    # 拼接文本和图像条件
    prompt_embeds = torch.cat([text_embeds, image_embeds], dim=1)

    # 扩散去噪循环（支持 CFG）
    for i, t in enumerate(scheduler.timesteps):
        model_pred = self.gen_transformer(hidden_states=latents, ...)
        # 范围内的 CFG
        guidance_scale = text_guidance_scale if cfg_range[0] <= i/total_steps <= cfg_range[1] else 1.0
        if guidance_scale > 1.0:
            model_pred = model_pred_uncond + guidance_scale * (model_pred - model_pred_uncond)

    # VAE 解码
    image = self.gen_vae.decode(latents, return_dict=False)[0]
    return OmniOutput(multimodal_outputs=image)
```

### 虚拟运行时信息

```python
def get_dummy_runtime_additional_information(self, num_reqs):
    # 为 CUDA graph 预热生成 dummy 输入
    return [{"text_prompt_embeds": ..., "image_prompt_embeds": ..., ...}]
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `MammothModa2DiTPipeline` | 类 | DiT 生成阶段管线 |

## 与其他模块的关系

- 使用 `mammothmoda2_dit_model.Transformer2DModel` 作为核心模型
- 使用自定义的 `FlowMatchEulerDiscreteScheduler`（来自 `schedulers.py`）
- 使用 `Mammothmoda2Config` 配置类（来自 `transformers_utils`）
- 返回 `OmniOutput` 以兼容 vLLM 的多模态输出接口
- 使用 `AutoWeightsLoader` 和 `WeightsMapper` 加载权重

## 总结

MammothModa2DiTPipeline 是一个非自回归的扩散生成阶段，接收上游 AR 模型产生的条件 token，通过 DiT + VAE 生成图像。其特色包括：CFG 范围控制（只在指定步数范围内应用 CFG）、可选的 Q-Former 图像条件精化、以及与 vLLM 原生接口的深度集成。
