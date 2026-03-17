# `pipeline_z_image.py` — Z-Image 图像生成管线

## 文件概述

该文件实现了 Z-Image 的文本到图像生成管线。管线使用 AutoModel 作为文本编码器、分布式 VAE (`DistributedAutoencoderKL`)、自定义的 Z-Image Transformer 和 Flow Matching Euler 调度器。支持 CFG 截断、CFG 归一化以及量化推理。

## 关键代码解析

### 分布式 VAE

```python
self.vae = DistributedAutoencoderKL.from_pretrained(
    model, subfolder="vae", local_files_only=local_files_only
).to(self._execution_device)
```

使用分布式 VAE 支持多 GPU 解码大分辨率图像。

### 量化支持

```python
quant_config = get_vllm_quant_config_for_layers(od_config.quantization_config)
self.transformer = ZImageTransformer2DModel(quant_config=quant_config)
```

### 文本编码 — 对话模板

```python
def _encode_prompt(self, prompt, ...):
    messages = [{"role": "user", "content": prompt_item}]
    prompt_item = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    # 编码并提取有效 token（通过 attention mask 过滤 padding）
    prompt_embeds = self.text_encoder(input_ids=text_input_ids, attention_mask=prompt_masks,
                                      output_hidden_states=True).hidden_states[-2]
    embeddings_list = [prompt_embeds[i][prompt_masks[i]] for i in range(len(prompt_embeds))]
```

使用倒数第二层隐藏状态，并按 attention mask 过滤出有效 token，返回 list（变长序列）。

### CFG 截断

```python
if self._cfg_truncation is not None and float(self._cfg_truncation) <= 1:
    if t_norm > self._cfg_truncation:
        current_guidance_scale = 0.0  # 后期步骤不应用 CFG
```

当归一化时间超过截断阈值后，停止应用 CFG，减少过度引导。

### CFG 归一化

```python
if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
    ori_pos_norm = torch.linalg.vector_norm(pos)
    new_pos_norm = torch.linalg.vector_norm(pred)
    max_new_norm = ori_pos_norm * float(self._cfg_normalization)
    scale = torch.where(new_pos_norm > max_new_norm, ...)
    pred = pred * scale
```

### 去噪循环

```python
for i, t in enumerate(timesteps):
    if apply_cfg:
        latent_model_input = latents_typed.repeat(2, 1, 1, 1)  # 正负 prompt 拼接
        prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds  # list 拼接
    latent_model_input = latent_model_input.unsqueeze(2)  # 添加帧维度
    model_out_list = self.transformer(
        latent_model_input_list, timestep_model_input, prompt_embeds_model_input,
    )[0]
    noise_pred = -noise_pred  # Z-Image 约定：输出取反
    latents = self.scheduler.step(noise_pred, t, latents)[0]
```

注意 Z-Image Transformer 接收 list 输入（每个元素是一个样本），且输出取反。

### 权重加载

```python
def load_weights(self, weights):
    loaded_weights = loader.load_weights(weights)
    # 手动标记 VAE 和 text_encoder 的参数为已加载
    loaded_weights |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
    loaded_weights |= {f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters()}
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ZImagePipeline` | 类 | 图像生成管线 |
| `get_post_process_func` | 函数 | 获取后处理函数 |
| `calculate_shift` | 函数 | 计算 timestep shift |
| `retrieve_timesteps` | 函数 | 获取调度器时间步 |

## 与其他模块的关系

- 使用 `ZImageTransformer2DModel` 作为核心去噪模型
- 使用 `DistributedAutoencoderKL` 进行分布式 VAE 解码
- 使用 `get_vllm_quant_config_for_layers` 支持量化
- 不使用 CFGParallelMixin（CFG 通过 batch 拼接实现）

## 总结

Z-Image 管线是一个功能完整的文本到图像管线，特色包括：(1) 分布式 VAE 解码；(2) 量化推理支持；(3) CFG 截断和归一化；(4) 预计算归一化时间步避免 GPU-CPU 同步。管线处理 Transformer 的 list 输入输出接口，这是 Z-Image 独有的设计。
