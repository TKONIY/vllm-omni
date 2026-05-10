# HunyuanImage3 AR + Diffusion 代码阅读指南

> 适用版本：`main @ 77480215`
> 用途：按照单条 T2I 请求的 forward 流程把代码走一遍，每一站给出 `file:line` 与"看什么"。
> 推荐配合 `vllm_omni/model_executor/stage_configs/hunyuan_image3_it2i_kv_reuse.yaml`（双卡 AR + 双卡 DiT + RDMA KV reuse）来对照——这是把"现状全部能跑通"的最小完整链路。

---

## 0. 文件地图（先扫一眼）

```
AR 侧（vllm-omni 标准模型路径，复用 vLLM 内核）
└── vllm_omni/model_executor/models/hunyuan_image3/
    ├── hunyuan_image3.py           # 主模型 + AR sampler + multimodal 处理（~2200 行）
    ├── autoencoder_kl_3d.py        # AR 用的 VAE encoder（IT2I 输入图）
    └── siglip2.py                  # AR 用的 ViT encoder（IT2I 输入图）

DiT 侧（vllm-omni diffusion 路径，独立 attention 实现）
└── vllm_omni/diffusion/models/hunyuan_image3/
    ├── pipeline_hunyuan_image3.py        # vllm-omni wrapper：HunyuanImage3Pipeline
    ├── hunyuan_image3_transformer.py     # backbone：HunyuanImage3Model + diffusers-style HunyuanImage3Text2ImagePipeline
    ├── hunyuan_image3_tokenizer.py       # DiT 自己的 tokenizer（会 re-encode AR 的 prompt）
    ├── autoencoder.py                    # DiT 用的 VAE decoder
    └── system_prompt.py                  # 系统提示模板

桥接 / 调度 / 配置
├── vllm_omni/model_executor/stage_input_processors/hunyuan_image3.py  # ar2diffusion 函数
├── vllm_omni/core/sched/omni_ar_scheduler.py                          # AR 调度器 + KV transfer 触发
├── vllm_omni/diffusion/diffusion_engine.py                            # DiT engine 入口
├── vllm_omni/diffusion/sched/{base_scheduler,request_scheduler,step_scheduler}.py
├── vllm_omni/diffusion/worker/diffusion_model_runner.py               # DiT runner
└── vllm_omni/engine/orchestrator.py                                   # 跨 stage 调度
```

入口示例：`examples/offline_inference/hunyuan_image3/end2end.py`。读完这份，回头再扫这个示例就一目了然。

---

## 1. 请求入口与跨 stage 编排

请求到达后由 orchestrator 决定先去哪个 stage：

- `vllm_omni/engine/orchestrator.py:218-260`：`Orchestrator._handle_initial_request` → `stage_pools[0].submit_initial`，request 进入 AR stage。
- `vllm_omni/engine/orchestrator.py:580-660`：`_route_output` 在每条 stage 输出后决定是否 `_forward_to_next_stage`。
- `vllm_omni/engine/orchestrator.py:765-850`：`_forward_to_next_stage` 是 AR→DiT 真正的桥；当 `next_pool.stage_type == "diffusion"` 时调用 `next_client.custom_process_input_func(...)`，它指向 `ar2diffusion`。

只看这三段就够了，剩下细节回头再翻。

---

## 2. AR stage forward：从 token 到 token

### 2.1 模型主类与初始化

入口：`hunyuan_image3.py:1395` `class HunyuanImage3ForConditionalGeneration`。

按顺序读 `__init__`（行 1426-1559）：

| 行 | 看什么 |
|---|---|
| 1445 | `self.model = HunyuanModel(...)`——backbone，继承自 vLLM `hunyuan_v1.HunYuanModel`，标准 paged attention |
| 1448 | `self.lm_head = ParallelLMHead(...)`——AR 输出头 |
| 1469-1486 | VAE / patch_embed / time_embed / vision_model / vision_aligner / timestep_emb——**注意**：这些是给 IT2I 的输入图编码用的，不是 DiT 输出头 |
| 1497-1512 | special token id 解析（`</think>`、`<recaption>`、`<answer>`、`<boi>`、`<img_size_*>`、`<img_ratio_*>` 等） |
| 1528-1537 | comprehension（I2T/T2T）模式的 token 屏蔽集合 |
| 1543-1553 | **stage transition forced sequence**：`</think>` → `[<recaption>]`，`</recaption>` → `[<answer>, <boi>, <img_size>]` |
| 1558-1559 | 替换 RoPE 为 `HunyuanImage3RotaryEmbedding`、把 MoE block 替换成 fp32 路由的版本 |

### 2.2 输入处理（multimodal）

只在 IT2I（输入图条件）路径需要，跳过即可看 T2I：

- `hunyuan_image3.py:1665` `_parse_and_validate_image_input`：从 `**kwargs` 解析输入图。
- `hunyuan_image3.py:1789` `_process_image_input`：返回 multimodal embeddings。
- `hunyuan_image3.py:1700-1820` 是图像→VAE/ViT 双通道→token embedding 的注入流程。

### 2.3 forward → compute_logits → sample（核心三步）

**forward**（`hunyuan_image3.py:1876`）：

```
input_ids → self.model(...) → hidden_states
```

`self.model` 是 vLLM `HunYuanModel`，会自己处理 chunked prefill、KV cache。看不到 attention 内部就对了——是 vLLM 内核。

**compute_logits**（`hunyuan_image3.py:1891`）：

```
hidden_states → self.lm_head → logits
```

**sample**（`hunyuan_image3.py:1913`）—— **重点**：

| 行 | 做什么 |
|---|---|
| 1926 | `assert logits.shape[0] == 1`——**当前 sampler 不是 batch-safe**，UAD 多请求要先解决 |
| 1942-1995 | 在 `<img_size_*>` 之后把 vocab 限定到 ratio token 集合，并做 greedy argmax |
| 2000+ | 应用 stage_transition：检测前文是否匹配触发 token，命中后强制 emit forced sequence 的下一个 token |

读这三个方法，AR forward 的全貌就清楚了。

### 2.4 mRoPE 位置编码（image-block token 顺序）

`hunyuan_image3.py:2034` `get_mrope_input_positions` → `hunyuan_image3.py:2098-2143` 处理 `<boi>` 后面的位置：

```
<boi>  <img_size>  <img_ratio>  <timestep:1×<img>>  <vae:K×<img>>  [<joint_img_sep>  <vit:M×<img>>]  <eoi>
```

**这是整个项目里 image-block token 的权威结构定义**，DiT 侧很多代码都默认这个布局。

---

## 3. AR sampling 触发 stage 切换：从 KV transfer 到 ar2diffusion

### 3.1 AR sampling 何时停

看 `hunyuan_image3_it2i_kv_reuse.yaml:37`：`stop_token_ids: [128025]  # <answer>`。结合 §2 的 `_stage_transitions[</recaption>]` 触发 `[<answer>, <boi>, <img_size>]`，**第一个 token `<answer>` 一出来 sampler 就 stop**——`<boi>/<img_size>/<img_ratio>` 实际不会被 AR 写下。这是 UAD 设计要打破的一个边界。

### 3.2 AR scheduler 触发 KV transfer

入口 `vllm_omni/core/sched/omni_ar_scheduler.py:43` `class OmniARScheduler`，按顺序读：

- 行 67：`self.kv_transfer_criteria = self._get_kv_transfer_criteria()`——读 yaml 的 `omni_kv_config.kv_transfer_criteria`。
- 行 120-190：`_process_kv_transfer_trigger`，两条触发分支：
  - `prefill_finished` → `moe.yaml` 用这条
  - `special_token` → 备用
  - 都没配（`it2i_kv_reuse.yaml` 就这样）→ 落到下一条
- 行 600-664：`_free_request`，请求 finish 时如果配了 `need_send_cache: true` 就触发 KV transfer。
- 行 671-715：`_mark_request_for_kv_transfer` 注册 `requests_needing_kv_transfer`。
- 行 233：`get_finished_requests_needing_kv_transfer` 把要传的请求带出 scheduler，下游 worker 真正发包。

### 3.3 KV 传输的发送端

`vllm_omni/worker/omni_connector_model_runner_mixin.py:961` `send_kv_cache` → `vllm_omni/distributed/omni_connectors/kv_transfer_manager.py:OmniKVTransferManager.handle_finished_requests_kv_transfer`。底层 `MooncakeTransferEngineConnector`（RDMA）。

### 3.4 ar2diffusion 桥接

入口：`vllm_omni/model_executor/stage_input_processors/hunyuan_image3.py:26` `def ar2diffusion(...)`。读这个函数（短，全文 ~117 行）：把 AR 输出的 token、生成文本、原始 prompt、目标 height/width、condition images 整理成 DiT pipeline 可消费的字典。

---

## 4. DiT stage forward：从请求到图

### 4.1 DiT engine 入口

`vllm_omni/diffusion/diffusion_engine.py:72` `class DiffusionEngine`，关注：

- 行 98-100：选 `RequestScheduler` 还是 `StepScheduler`——HunyuanImage3 走 `RequestScheduler`（pipeline 没声明 `supports_step_execution`）。
- 行 103-106：非 stepwise 强制 `max_num_running_reqs=1`。
- 行 115：`execute_fn = executor.execute_request`——整段 denoise loop 在 pipeline 里跑完才返回。
- 行 139：`step(request)`——单 request 入口；`async_add_req_and_wait_for_response` 把请求塞队列，worker thread 调 `pipeline.forward(req)`。

### 4.2 vllm-omni wrapper：HunyuanImage3Pipeline

`pipeline_hunyuan_image3.py:300` `class HunyuanImage3Pipeline`。

按顺序看 `__init__`（行 327-385）：

| 行 | 看什么 |
|---|---|
| 343 | `os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"`——**强制 SDPA**，注释解释是 mixed causal/full mask 的原因 |
| 348 | `self.model = HunyuanImage3Model(...)`——DiT backbone（独立实现，不是 vLLM 的） |
| 350-378 | VAE / vision / patch_embed / time_embed / final_layer / time_embed_2 / lm_head——**注意 lm_head 也在 DiT 侧加载**，gen_text 模式留有完整入口 |

### 4.3 DiT 入口 forward

`pipeline_hunyuan_image3.py:1329` `def forward(self, req: OmniDiffusionRequest, ...)`。读 1329-1430，关键：

| 行 | 看什么 |
|---|---|
| 1402 | `self._extract_ar_kv_from_request(req)`——把 AR 传过来的 `request.sampling_params.past_key_values` 拆成 layer-wise 字典 |
| 1404-1413 | `self.prepare_model_inputs(mode="gen_image", ...)` ——构造 input_ids / position_ids / attention_mask / cond_* / system_prompt |

### 4.4 prepare_model_inputs（输入打包）

`pipeline_hunyuan_image3.py:746-927`，关注：

- 行 749：默认 `mode="gen_image"`。
- 行 780-810：分支处理 IT2I（条件图）vs T2I。
- 行 822-925：构造 `model_input_kwargs`，包含 `input_ids / position_ids / past_key_values=None / custom_pos_emb / mode / image_mask / gen_timestep_scatter_index / cond_*` 等。

### 4.5 attention mask 构造

`pipeline_hunyuan_image3.py:949-969` `_prepare_attention_mask_for_generation`：

- 构造 `(bsz, 1, seq_len, seq_len)` 4D bool mask；
- 文本部分下三角因果；
- image_slice 区域全 1（image-block 内部全连接）。

这是为什么需要 SDPA：vLLM/FlashAttention 没法在一次 kernel 里表达"causal + 局部 full"的 mixed mask。

---

## 5. 真正的 denoise loop：HunyuanImage3Text2ImagePipeline

DiT 的真正主循环不在 `HunyuanImage3Pipeline.forward`（vllm-omni wrapper），而在另一个 diffusers-style pipeline：

**`hunyuan_image3_transformer.py:2420` `class HunyuanImage3Text2ImagePipeline(DiffusionPipeline)`**

这是从 Tencent reference 移植过来的扩散主循环。`HunyuanImage3Pipeline.forward` 内部会走到这里。按 `__call__`（行 2814）顺序读：

### 5.1 准备阶段（2890-2961）

| 行 | 看什么 |
|---|---|
| 2900 | 检测 `cfg_parallel_ready`（CFG 并行需要 world_size==2） |
| 2906 | `retrieve_timesteps`——flow-match scheduler 设定 timesteps |
| 2915 | `prepare_latents`——随机初始化 noisy latent |
| 2932 | `_prepare_attention_mask_for_generation`——同 §4.5 |
| 2939-2951 | CFG parallel split：rank0 拿 conditioned、rank1 拿 unconditioned |
| 2956-2960 | 设定 `query_lens`、`seq_lens`（**所有请求同长度**——`ImageKVCacheManager.__call__` 行 1011 会 assert） |

### 5.2 AR KV reuse 处理（2965 入口）

`_maybe_handle_ar_kv_reuse`（行 2766）做 4 件事：

1. **算复用长度**（行 2780）`_get_kv_reuse_len(行 2631)`——读 tokenizer 输出的 `think_recaption_end_pos`，**注意行 2637**：`pos_reuse_kv_len = think_recaption_end_pos[0][0]`，注释明言"not reuse the last token `</think>` or `<recaption>`"，所以边界 token 本身**不复用**。
2. **inject positive KV**（行 2789）`self.model.inject_ar_kv_into_layers(ar_kv_data, positive_reuse_len)`，跳到 `pipeline_hunyuan_image3.py:1290`：把 AR KV 写到每层 `layer.self_attn.image_attn._injected_ar_kv`。
3. **CFG negative prefill**（行 2792-2802）：若开 CFG，对 `[negative_reuse_len, positive_reuse_len)` 这段跑一次 `forward_call(uncond_cfg_prefill=True, mode="gen_image", num_image_tokens=0)` 把 negative branch 的 KV 也补上。看 `_build_negative_cfg_prefill_inputs`（行 2691）和 `_build_neg_ar_kv`（`hunyuan_image3_transformer.py:967`）协同。
4. **截断已注入的部分**（行 2805-2810）`_truncate_reused_prefix`（行 2740）：把 input_ids、attention_mask、position_ids 的前 `positive_reuse_len` 切掉，剩余给 first_step 真正算的部分。

### 5.3 主循环：每步去噪（2982-3070）

```python
for i, t in enumerate(timesteps):
    # 1. 准备 latent
    latent_model_input = (latents 或 cat([latents]*2))    # CFG 决定
    t_expand = t.repeat(...)
    
    # 2. forward
    model_inputs = self.model.prepare_inputs_for_generation(input_ids, images=latent_model_input, timestep=t_expand, ...)
    model_output = self.model.forward_call(**model_inputs, first_step=(i == 0))
    pred = model_output["diffusion_prediction"]
    
    # 3. CFG 合成
    if cfg_parallel_ready:
        gathered = cfg_group.all_gather(pred, separate_tensors=True)
        pred = self.cfg_operator(gathered[0], gathered[1], guidance_scale, step=i)
    elif self.do_classifier_free_guidance:
        pred_cond, pred_uncond = pred.chunk(2)
        pred = self.cfg_operator(pred_cond, pred_uncond, guidance_scale, step=i)
    
    # 4. scheduler step：noise → cleaner latent
    latents = self.scheduler.step(pred, t, latents, ...)[0]
    
    # 5. 准备下一步输入（更新 input_ids / position_ids / attention_mask）
    if i != len(timesteps) - 1:
        model_kwargs = self.model._update_model_kwargs_for_generation(model_output, model_kwargs)
```

行号定位：

| 行 | 操作 |
|---|---|
| 2989 | non-CFG-parallel 时 batch 翻倍 |
| 3012-3017 | `prepare_inputs_for_generation`（`pipeline_hunyuan_image3.py:951`）按 `first_step` 选 input_ids vs inputs_embeds |
| 3020 | **forward_call**——见 §6 |
| 3022 | 拿到 `diffusion_prediction` |
| 3034 / 3037 | CFG 合成 |
| 3040 | `self.scheduler.step(...)` 推 latents |
| 3043-3056 | 更新下一步的 model_kwargs（image_mask / position_ids 偏移） |

### 5.4 VAE decode（3072-3093）

```
latents = latents / vae.config.scaling_factor + shift_factor
image = self.vae.decode(latents).sample
image = self.image_processor.postprocess(image, output_type="pil")
```

---

## 6. 模型 forward_call：每步真正的算子流

入口：`pipeline_hunyuan_image3.py:1129` `def forward_call(...)`，读到 1288。

注意：`forward_call` 在 vllm-omni wrapper（`HunyuanImage3Pipeline`）上，**不**在 `HunyuanImage3Model` 上。`self.model.forward_call(...)` 调用的 `self.model` 在 `Text2ImagePipeline.__init__` 里被注册成 `HunyuanImage3Pipeline` 实例（不是单纯的 backbone），所以这里 `self.model.forward_call` 跳到 wrapper 上。

### 6.1 输入嵌入分叉（1180-1207）

```python
if mode == "gen_text":              # vllm-omni 当前路径不会走这里（见 §9.1）
    gen_timestep_scatter_index = None
    token_h, token_w = None, None
else:                                # gen_image
    if first_step:
        inputs_embeds, token_h, token_w = self.instantiate_vae_image_tokens(
            inputs_embeds, images, timestep, image_mask)             # 把 placeholder <img> 替换成 VAE patch embed
        inputs_embeds = self.instantiate_timestep_tokens(
            inputs_embeds, timestep, gen_timestep_scatter_index)     # 把 <timestep> placeholder 替换成 timestep emb
    else:
        t_emb = self.time_embed(timestep)                            # 不再嵌入 prompt prefix
        image_emb, _, _ = self.patch_embed(images, t_emb)
        timestep_emb = self.timestep_emb(timestep).reshape(bsz, -1, n_embd)
        inputs_embeds = torch.cat([timestep_emb, image_emb], dim=1)
```

`first_step` 与后续 step 的差异是**整个 chunked-prefill 直觉的源头**：first_step embed 全序列，后续 step 只 embed image-block，prompt KV 靠层内 `image_kv_cache_map` 复用。

### 6.2 调用 backbone（1211-1230）

```python
with set_forward_context(None, self.vllm_config):
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        ...
        mode=mode,
        first_step=first_step,
        query_lens=query_lens,
        seq_lens=seq_lens,
        num_image_tokens=num_image_tokens,
        gen_timestep_scatter_index=gen_timestep_scatter_index,
    )
```

这个 `self.model` 才是 `HunyuanImage3Model`（`hunyuan_image3_transformer.py:1887`）。继续往下看。

### 6.3 输出头分叉（1232-1259）

```python
if mode == "gen_text":
    hidden_states = self.model.ln_f(hidden_states)
    logits = self.lm_head(hidden_states)
    diffusion_prediction = None
else:                                # gen_image
    diffusion_prediction = self.ragged_final_layer(
        hidden_states, image_mask, timestep, token_h, token_w, first_step)
```

`ragged_final_layer`（`pipeline_hunyuan_image3.py:590`）只对 image_mask 标记的 token 跑 final layer，其他位置不算——这是"ragged"的来源。

---

## 7. backbone 内部：HunyuanImage3Model

`hunyuan_image3_transformer.py:1887` `class HunyuanImage3Model(nn.Module)`。它和 AR 侧的 `HunyuanModel` **不是同一个类**。

读 `forward`（行 2230）：

| 行 | 操作 |
|---|---|
| 2258-2262 | `inputs_embeds → hidden_states` |
| 2273-2333 | SP（sequence parallel）路径：把 hidden_states 切成 text/image 两段（`pre_processor`）；如果开 ulysses 还要拼 padding |
| 2335-2357 | for layer in self.layers: 调用 `HunyuanImage3DecoderLayer.forward(...)`，传 mode/first_step 等 kwargs |
| 2370-2376 | SP 路径：post_processor 把 image 段 gather 回原 dim |

每一层 `HunyuanImage3DecoderLayer`（行 1682）的 forward（行 1738）：input_layernorm → self_attn → residual → post_attention_layernorm → MLP/MoE → residual。

### 7.1 self_attn = HunYuanAttention（核心）

`hunyuan_image3_transformer.py:1530` `class HunYuanAttention`，看 `__init__`（1545-1636）：

| 行 | 看什么 |
|---|---|
| 1601-1606 | `get_rope(...)` 标准 RoPE |
| 1609-1618 | `self.attn = Attention(causal=False, ...)`——这里的 `Attention` 是 `vllm_omni/diffusion/attention/layer.Attention`（SDPA / Ring），**不是 vLLM paged**。`causal=False` 是因为 dense mask 已经把因果体现在 mask 里 |
| 1621-1627 | `self.image_attn = ImageKVCacheManager(...)`——gen_image 模式专用，§8 详读 |
| 1628-1632 | `self.image_rope2d_emb = HunYuanRotary2DEmbedder(...)`——image token 的 2D RoPE |

`forward`（1638-1679）：

```python
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split(...)

# HF 风格 Cache（DiT 当前不用 vLLM paged 路径）
past_key_value = kwargs.get("past_key_value")
if past_key_value is not None:
    k, v = past_key_value.update(k, v, self.layer_id, ...)

# RoPE 选哪种取决于 mode
if mode == "gen_image":
    q, k = self.image_rope2d_emb(q, k, hidden_states, custom_pos_emb, **kwargs)
else:
    q, k = self.rotary_emb(positions, q, k)

# attention 选哪种取决于 mode
if mode == "gen_image":
    attn_output = self.image_attn(q, k, v, attention_mask=attention_mask, **kwargs)   # 进入 §8
else:
    attn_output = self.attn(q, k, v)                                                   # 普通 SDPA / Ring
```

---

## 8. ImageKVCacheManager（DiT 内最关键的一段）

入口：`hunyuan_image3_transformer.py:848` `class ImageKVCacheManager`（**注意它不是 nn.Module，是普通 class**）。

### 8.1 状态字段

`__init__`（853-877）：

| 行 | 字段 | 含义 |
|---|---|---|
| 866 | `image_kv_cache_map: tuple[Tensor, Tensor] \| None` | 缓存的 prompt KV（first_step 写入，后续 step 复用） |
| 867 | `_injected_ar_kv: list[tuple] \| None` | 从 AR 侧注入的 prefix KV，bs=1 时是 `[(k,v)]`，CFG bs=2 时是 `[(pos),(neg)]` |
| 871-877 | `self.attn = Attention(causal=False, ...)` | 这一层用的 SDPA-based attention 实例 |

### 8.2 `__call__`（行 993-1071）—— 主入口

按 if/elif 顺序读：

```python
if uncond_cfg_prefill:
    # CFG 第二次 prefill：构造 negative AR KV
    key, value = self._build_neg_ar_kv(key, value, seq_len)   # 行 967

elif first_step:
    # 第一步：吃掉 _injected_ar_kv，把 [ar_kv | new_prompt_kv] 拼起来缓存
    self.image_kv_cache_map = None                            # 行 1032 防御性 reset
    key, value = self._cache_prompt_kv(key, value, seq_len)   # 行 879

else:
    # 后续步：从 image_kv_cache_map 回放 prompt KV，拼上当前 image-block KV + zero_eoi
    key, value = self._reuse_prompt_kv(key, value, seq_len, bs)   # 行 922
```

### 8.3 三个核心方法

`_cache_prompt_kv`（879-920）：

```
ar_kv (前缀)  ⊕  key (当前 step 的 prompt+image+eoi 投影)
              ↓
[ar_kv_part | prompt_part | image_part | eoi_part]
              ↓ 取前 (seq_len - image_size - 1) 段
image_kv_cache_map = (cached_key, cached_value)        # 缓存"prompt 部分"
return 完整拼好的 (key, value)                            # 当前 step 仍用整段
```

`_reuse_prompt_kv`（922-965）：

```
image_kv_cache_map (上一次的 cached prompt)
                 ⊕
key (当前 step 只算了 image-block 的投影，shape: [bs, q_len=image_token_len, ...])
                 ⊕
zero_eoi (一个全 0 token，对应 <eoi>)
                 ↓
[cached_prompt | new_image | zero_eoi]
                 ↓
返回这个完整 tensor 给 self.attn(q, k, v)
```

`_build_neg_ar_kv`（967-991）：

```
当 negative AR KV 比 positive 短 → 共享 positive 的前缀
[positive_ar[:negative_reuse_len] | negative_prefill_tokens] = neg_ar_kv
self._injected_ar_kv = [(pos), (neg)]
```

### 8.4 SP 路径（1024-1067）

`if self.sp_size > 1` 时把 prompt 段（`joint_text`）和 image 段拆开喂给 SDPA，理由是 sequence parallel 把 image token shard 到多 rank。一般首读跳过。

### 8.5 关键不变量

- **每个 layer 一个 ImageKVCacheManager 实例**，状态是 instance 字段。
- 多请求并发会互相覆盖（§4.5 of `design.md`）。
- 当前能跑通是因为 `RequestScheduler` 串行 + `max_batch_size=1`。
- `__call__:1011` 的 `assert query.shape[0] == bs * q_len` 隐式要求 batch 内 q_len 全相等。

---

## 9. 易混点和必读 sidebar

### 9.1 DiT 侧的 gen_text 分支是死代码

`forward_call` / `HunYuanAttention.forward` 都有 `if mode == "gen_text"` 分支，但 vllm-omni 当前调用方（`HunyuanImage3Pipeline.forward` 行 1408、`prepare_model_inputs` 行 749、`_build_negative_cfg_prefill_inputs` 行 2721）**全部传 `mode="gen_image"`**。`gen_text` 分支是从 Tencent reference 模型移植过来的——reference 是单一 transformer 同时支持 AR 和 diffusion，vllm-omni 把 AR 拆出来跑 vLLM 标准实现，这个分支就此闲置。

读代码遇到 `if mode == "gen_text"` 时知道**当前路径不会走**就行；UAD 设计反而要把它复活。

### 9.2 两个"pipeline"同名易混

| 类 | 文件:行 | 角色 |
|---|---|---|
| `HunyuanImage3Pipeline` | `pipeline_hunyuan_image3.py:300` | vllm-omni wrapper，DiffusionEngine 直接调用，提供 `forward(req)` |
| `HunyuanImage3Text2ImagePipeline` | `hunyuan_image3_transformer.py:2420` | diffusers-style 内层 pipeline，承担 `__call__` 主循环和 AR KV reuse |

vllm-omni wrapper 在 `forward` 里调内层 pipeline（`hunyuan_image3_transformer.py:3020` 也是 `self.model.forward_call(...)`，这里的 `self.model` 是 vllm-omni wrapper）。看 callstack 时注意区分。

### 9.3 两份独立 backbone

- AR 侧 backbone：vLLM `HunYuanModel`（vllm/hunyuan_v1.py），用 vLLM paged Attention。
- DiT 侧 backbone：`HunyuanImage3Model`（`hunyuan_image3_transformer.py:1887`），用 `vllm_omni/diffusion/attention/layer.Attention`（SDPA + Ring）。

两边 checkpoint 兼容（同一份权重经过各自的 unexpected_keywords 过滤），但实例是物理上两份。读到 `from vllm.model_executor.models.hunyuan_v1 import HunYuanModel`（`hunyuan_image3.py:55-61`）和 `class HunyuanImage3Model(nn.Module)`（`hunyuan_image3_transformer.py:1887`）时，要意识到这是两条平行路径。

### 9.4 attention 类同名易混

- `vllm.model_executor.layers.attention.Attention`：vLLM paged。
- `vllm_omni.diffusion.attention.layer.Attention`：SDPA / Ring，DiT 用。
- `ImageKVCacheManager.attn`：内嵌一个 `vllm_omni.diffusion.attention.layer.Attention(causal=False)` 实例。

读到 `Attention(...)` 时先看 import。

### 9.5 token 顺序的"权威定义"

读到 image-block 相关代码时，把这个挂在脑里：

```
<boi> <img_size> <img_ratio> <timestep:1×<img>> <vae:K×<img>> [<joint_img_sep> <vit:M×<img>>] <eoi>
```

定义见 `hunyuan_image3.py:2098-2143`。`timestep` 是 1 个 `<img>` placeholder（占位），不是 timestep token 本身——timestep 信息在 `instantiate_timestep_tokens`（`pipeline_hunyuan_image3.py:541`）里以 embedding 形式注入。

---

## 10. 推荐阅读顺序（一个小时跑完一遍）

1. **总览（10 min）**：本指南 §0 + §1 + 跑一遍 `examples/offline_inference/hunyuan_image3/end2end.py` 看输出。
2. **AR 半小时**：
   - `hunyuan_image3.py:1395` 类头 → `__init__` → `forward` → `compute_logits` → `sample`。
   - 跳读 stage transition（行 1543-1553）。
   - **跳过** multimodal/VAE/ViT 这些 IT2I 输入处理。
3. **bridge 5 min**：
   - `omni_ar_scheduler.py` 行 600-664 + 行 120-190。
   - `stage_input_processors/hunyuan_image3.py:26` `ar2diffusion`。
4. **DiT wrapper 10 min**：
   - `pipeline_hunyuan_image3.py:300` `__init__` → `forward(req)` → `prepare_model_inputs`。
   - **跳过** `forward_call` 内部细节，先看 wrapper 全貌。
5. **DiT 主循环 15 min**：
   - `hunyuan_image3_transformer.py:2814` `__call__` 整段。
   - 重点 §5.2（KV reuse）和 §5.3（denoise loop）。
6. **forward_call + ImageKVCacheManager 15 min**：
   - `pipeline_hunyuan_image3.py:1129` `forward_call`。
   - `hunyuan_image3_transformer.py:848` `ImageKVCacheManager` 三个方法。

读到这里，HunyuanImage3 在 vllm-omni 里的 forward 全图就清楚了。剩下 SP / CFG-parallel / TeaCache 都是侧支，按需查。
