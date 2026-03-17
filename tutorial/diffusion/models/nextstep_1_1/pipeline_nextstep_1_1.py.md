# `pipeline_nextstep_1_1.py` — NextStep-1.1 图像生成管线

## 文件概述

该文件实现了 NextStep-1.1 的完整图像生成管线，采用自回归 Flow Matching 方式逐 token 生成图像。与标准扩散管线不同，NextStep 使用 LLM 骨干进行自回归解码，每步通过 FlowMatchingHead 采样一个图像 token。支持 CFG 并行推理。

## 关键代码解析

### 管线初始化

```python
class NextStep11Pipeline(nn.Module):
    def __init__(self, *, od_config, prefix=""):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, ...)
        config = NextStepConfig.from_json(os.path.join(model_path, "config.json"))
        self.model = NextStepModel(config)
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.down_factor = vae_factor * latent_patch_size  # VAE + patch 总下采样倍数
```

### 自回归解码

```python
def decoding(self, c, attention_mask, past_key_values, max_new_len, ...):
    for step in indices:
        if cfg_parallel:
            # CFG 并行：每个 rank 处理一个 CFG 分支
            c_proj = self.model.image_out_projector(c)
            c_gathered = cfg_group.all_gather(c_proj, separate_tensors=True)
            c_full = torch.cat(c_gathered, dim=0)
            if cfg_rank == 0:
                token_sampled = self.model.image_head.sample(c=c_full.squeeze(1), ...)
            cfg_group.broadcast(token_sampled, src=0)
        else:
            token_sampled = self.model.image_head.sample(c=c_proj.squeeze(1), ...)

        # 投影回 LLM 隐藏空间
        cur_inputs_embeds = self.model.image_in_projector(tokens[:, -1:])
        # LLM 前向一步
        outputs = self.model.forward_model(inputs_embeds=cur_inputs_embeds, ...)
        c = outputs.last_hidden_state[:, -1:]
```

### CFG 并行优化

```python
if cfg_parallel:
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_group = get_cfg_group()
    # 分割 KV 缓存给每个 rank
    batch_per_rank = full_bsz // cfg_mult
    start = cfg_rank * batch_per_rank
    # 重建 StaticCache 给当前 rank
    new_cache = StaticCache(config=self.config, max_cache_len=...)
    for layer_idx, layer in enumerate(old_cache.layers):
        new_layer.keys.copy_(layer.keys[start:end])
```

### 完整生成流程

```python
def forward(self, req, ...):
    # 1. 构建 CFG 文案
    captions, images, cfg_mult, effective_cfg_img = self._build_captions(...)
    # 2. 分词 + 添加前缀 ID
    input_ids, attention_mask = self._add_prefix_ids(hw, input_ids, attention_mask)
    # 3. LLM prefill
    inputs_embeds = self.model.prepare_inputs_embeds(input_ids, latents)
    outputs = self.model.forward_model(inputs_embeds=inputs_embeds, ...)
    # 4. 自回归解码
    tokens = self.decoding(c=outputs.last_hidden_state[:, -1:], ...)
    # 5. Unpatchify + VAE 解码
    latents = self.model.unpatchify(tokens)
    sampled_images = self.vae.decode(latents).sample
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `NextStep11Pipeline` | 类 | 主管线 |
| `get_nextstep11_post_process_func` | 函数 | 获取后处理函数 |

## 与其他模块的关系

- 使用 `NextStepModel` 作为 LLM 骨干
- 使用 `AutoencoderKL`（自定义）作为 VAE
- 使用 `StaticCache` 管理 KV 缓存
- 使用 CFG 并行组进行分布式 CFG

## 总结

NextStep-1.1 管线采用独特的自回归 Flow Matching 架构：LLM 骨干逐步生成图像 token，每步通过 FlowMatchingHead 进行 SDE 采样。其 CFG 并行实现特别精巧，通过分割 StaticCache 和 all_gather/broadcast 在多 GPU 间高效地协调 CFG 分支。
