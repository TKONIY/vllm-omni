# `cfg_parallel.py` — QwenImage CFG 并行 Mixin

## 文件概述

本文件实现了 QwenImage 系列所有管线共享的 CFG 并行 Mixin `QwenImageCFGParallelMixin`。它封装了扩散循环（diffuse 方法）和 CFG 并行配置校验，被 `QwenImagePipeline`、`QwenImageEditPipeline`、`QwenImageEditPlusPipeline` 和 `QwenImageLayeredPipeline` 四种管线继承。

## 关键代码解析

### 1. 扩散循环

```python
class QwenImageCFGParallelMixin(CFGParallelMixin, ProgressBarMixin):
    def diffuse(self, prompt_embeds, ..., timesteps, do_true_cfg, guidance, true_cfg_scale,
                image_latents=None, cfg_normalize=True, additional_transformer_kwargs=None):
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(device=latents.device)

            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)

            positive_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep / 1000,
                "encoder_hidden_states": prompt_embeds, ...
            }

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize, output_slice
            )
            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
```

关键设计点：
- 时间步除以 1000 归一化到 [0, 1] 范围
- 编辑管线通过拼接 `image_latents` 将条件图像传入
- `output_slice` 用于从拼接输出中截取噪声预测部分

### 2. CFG 并行校验

```python
def check_cfg_parallel_validity(self, true_cfg_scale, has_neg_prompt):
    if get_classifier_free_guidance_world_size() == 1:
        return True
    if true_cfg_scale <= 1:
        logger.warning("CFG parallel is NOT working correctly when true_cfg_scale <= 1.")
        return False
    if not has_neg_prompt:
        logger.warning("CFG parallel is NOT working correctly when there is no negative prompt.")
        return False
    return True
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `QwenImageCFGParallelMixin` | 类 | CFG 并行 Mixin |
| `diffuse` | 方法 | 扩散去噪循环 |
| `check_cfg_parallel_validity` | 方法 | CFG 并行配置校验 |

## 与其他模块的关系

- **`CFGParallelMixin`**：继承自 `vllm_omni.diffusion.distributed.cfg_parallel`
- **`ProgressBarMixin`**：进度条支持
- 被 `pipeline_qwen_image.py`、`pipeline_qwen_image_edit.py`、`pipeline_qwen_image_edit_plus.py`、`pipeline_qwen_image_layered.py` 四个管线继承

## 总结

`cfg_parallel.py` 提供了 QwenImage 系列管线共享的扩散循环实现，通过 `predict_noise_maybe_with_cfg` 和 `scheduler_step_maybe_with_cfg` 自动处理 CFG 并行的通信和同步，包含编辑管线的条件图像拼接支持。
