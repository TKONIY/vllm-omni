# DreamZero Review

## 结论

- `action_encoder.py` 与 DreamZero 原始实现一致。
- `causal_wan_model.py` 的 inference 主路径已经和 DreamZero 对齐，并且高优先级的“热点路径未复用 vllm-omni 基础设施”问题已解决。
- 当前已完成单卡与 TP=2 的 GPU 精度对齐；TP=2 下 `DistributedRMSNorm`、T2V/I2V cross-attention、self-attention、attention block、以及 tiny full model 的 prefill + 一步 AR step 均已通过和 DreamZero 的逐项比对。
- 但当前实现仍不能判定为“与 DreamZero 完全一致”。剩余差异主要是并行执行框架未接入、`img_emb` 创建条件不同、以及缺少 `init_weights()` 裸模型初始化路径。
- 训练路径已经按要求删除；相关缺失不再计为问题。

## 检查通过项

### `action_encoder.py` 对齐通过

- `swish`、`SinusoidalPositionalEncoding` 对齐 `groot/vla/model/n1_5/modules/action_encoder.py`。
- `CategorySpecificLinear`、`CategorySpecificMLP`、`MultiEmbodimentActionEncoder` 对齐 `wan_video_dit_action_casual_chunk.py:31-90`。
- GPU 环境下复测通过：`MASTER_PORT=29601 PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest tests/dreamzero/test_action_encoder.py -v -s`。
- 实测结果为 bit-identical / `max_diff=0.00e+00`。

### `causal_wan_model.py` inference 主路径对齐通过

- 静态对比与 DreamZero 一致或等价：
  - `sinusoidal_embedding_1d`、`rope_params`、`rope_apply`、`rope_action_apply`、`causal_rope_action_apply`
  - `CausalHead`
  - `_create_freqs`
  - `unpatchify`
  - `_forward_blocks`
  - `_forward_inference`
  - `CausalWanSelfAttention` 的 KV-cache 推理分支
- `WanT2VCrossAttention` 保留了 DreamZero 的 `context_lens` 接口形状兼容性，同时底层仍走 vllm-omni `Attention`。
- `cross_attn_norm` 的模型默认值保持为 DreamZero 原始构造参数默认值 `True`，block 默认值仍为原始的 `False`。

### 高优先级问题已解决：热点路径已切到 vllm-omni 基础设施

- `patch_embedding` 已切换为 `Conv3dLayer`，见 `/home/yangshen/code/vllm-omni-wm/vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:733-739`。
- self-attention 的 `q/k/v/o` 已切换为 `ColumnParallelLinear` / `RowParallelLinear`，QK norm 改为 TP-aware `DistributedRMSNorm`，见 `/home/yangshen/code/vllm-omni-wm/vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:427-440`。
- T2V / I2V cross-attention 的 `q/k/v/o` 与 `k_img/v_img` 已切到 `ColumnParallelLinear` / `RowParallelLinear`，见 `/home/yangshen/code/vllm-omni-wm/vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:267-374`。
- FFN 已切到 `ColumnParallelLinear -> GELU -> RowParallelLinear`，见 `/home/yangshen/code/vllm-omni-wm/vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:567-571`。
- `MLPProj` 已继续复用 `ColumnParallelLinear + RowParallelLinear`，见 `/home/yangshen/code/vllm-omni-wm/vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:221-244`。

### GPU 精度验证通过

- 运行命令：`PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest tests/dreamzero/test_causal_wan_model.py -v -s`。
- 共 7 项测试全部通过，覆盖：
  - 热点层类型检查
  - `DistributedRMSNorm` 精度
  - T2V / I2V cross-attention 精度
  - self-attention 精度
  - attention block 精度
  - tiny full model 的 prefill + 一步 AR step 精度
- 关键结果：
  - `CausalWanModel.prefill.video` `max_diff=0.000e+00`
  - `CausalWanModel.step.video` `max_diff=0.000e+00`
  - `CausalWanModel.step.action` `max_diff=1.397e-09`
  - layer KV cache 对齐误差在 `7.153e-07` 以内

### TP=2 GPU 精度验证通过

- 运行命令：`PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest tests/dreamzero/test_causal_wan_model_tp2.py -v -s`。
- `tests/dreamzero/test_causal_wan_model_tp2.py` 已覆盖：
  - 热点层 shard/type 检查
  - `DistributedRMSNorm` TP 精度
  - T2V / I2V cross-attention TP 精度
  - self-attention TP 精度
  - attention block TP 精度
  - tiny full model 的 prefill + 一步 AR step TP 精度
- 实测结果：`1 passed`，TP=2 全量精度校验通过。

### 新增修复记录：TP=2 I2V 精度偏差

- 问题现象：`WanI2VCrossAttention.prefill` 在 TP=2 下与 DreamZero 出现明显偏差，单卡路径正常。
- 定位结果：
  - `q/k/v/k_img/v_img` 的分片线性投影与 DreamZero 一致；
  - `self.o` 单独喂入正确的 `pre_o` 时输出也与 DreamZero 一致；
  - 真正的偏差出现在第一次 cross-attention 之后再次执行 `k_img -> norm_k_img` 时；
  - `k_img` 线性输出本身正确，但 `DistributedRMSNorm` 在 TP=2 下的输出错误。
- 根因：
  - vLLM 的 TP all-reduce 默认走 pynccl 当前 stream；在 `initialize_model_parallel(...)` 之后，实际 CUDA current stream 与 vLLM 线程本地记录的 stream 存在分离。
  - 对 `DistributedRMSNorm` 这类“本地计算 + TP all-reduce + 后续本地计算”的小张量路径，这种 stream 分离会导致规约与本地算子之间出现竞态，从而在 I2V 的第二次 attention 前污染 `norm_k_img` 输出。
- 修复：
  - `DistributedRMSNorm` 改为直接在 TP group 上执行 `torch.distributed.all_reduce(...)`，避免走 pynccl 的 TLS stream 路径，见 `vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:196`。
  - `WanI2VCrossAttention.forward` 在第一次 attention 后、`k_img/v_img` 前，以及输出投影 `self.o` 前，显式恢复到默认 CUDA stream，见 `vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:366`。
- 结论：
  - 修复后，I2V TP=2 精度恢复与 DreamZero 一致；
  - 单卡回归测试也保持通过。

## 剩余问题

### [中] 还没有接入 WAN2.2 / Bagel 现成的 sequence parallel / ring parallel 框架

- 当前 DreamZero 端口虽然复用了 `vllm_omni.diffusion.attention.layer.Attention`，但所有注意力都显式设置了 `skip_sequence_parallel=True`，见 `/home/yangshen/code/vllm-omni-wm/vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:279-284`、`/home/yangshen/code/vllm-omni-wm/vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:341-347`、`/home/yangshen/code/vllm-omni-wm/vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:433-440`。
- 模型本身也没有像 WAN2.2 一样声明 `_sp_plan`，因此没有进入 vllm-omni 现成的 SP / Ulysses / Ring 执行框架。
- 这不影响当前单卡 GPU 精度对齐，但意味着“并行策略支持”仍未达到 Bagel / WAN2.2 的复用程度。

### [低] `img_emb` 创建条件与原始 DreamZero 不一致

- 当前在 `model_type in ("i2v", "ti2v")` 时创建 `img_emb`，见 `/home/yangshen/code/vllm-omni-wm/vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py:771-777`。
- 原始 DreamZero 只在 `model_type == "i2v"` 时创建，见 `/home/yangshen/code/dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py:1380-1381`。
- 当前已验证的 `t2v` tiny 路径不受影响，但严格说对象布局仍非完全一致。

### [低] 缺少与原始 DreamZero 完全一致的 `init_weights()` 裸模型初始化路径

- 原始 DreamZero 会在构造函数末尾调用 `init_weights()`，见 `/home/yangshen/code/dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py:1383-1387`。
- 当前 vllm-omni 版本没有保留这一初始化流程。
- 如果运行时总是加载转换后的权重，这个差异影响很小；但从“空模型构造行为”角度看，仍非 100% 一致。

## 不计为问题的差异

- `_forward_train` 缺失：按你的要求，训练路径应当删除，不再计为缺陷。
- TensorRT 专用入口 `_forward_inference_trt` / `_forward_inference_trt_droid` 未保留：本次 inference-only 端口不作为阻塞项。
- `gradient_checkpointing`、`ModelMixin` / `ConfigMixin` 等训练侧能力未保留：不视为本次 review 的阻塞项。

## 验证记录

- 通过：`MASTER_PORT=29601 PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest tests/dreamzero/test_action_encoder.py -v -s`
- 通过：`PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest tests/dreamzero/test_causal_wan_model.py -v -s`
- 通过：`PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest tests/dreamzero/test_causal_wan_model_tp2.py -v -s`
- 备注：`tests/dreamzero/conftest.py` 目前把 `MASTER_PORT` 固定成 `29599` 的默认值，若上一次测试残留监听端口，占用时需要通过环境变量覆盖端口。
