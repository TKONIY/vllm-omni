# DreamZero 推理调用链梳理

## 原始 DreamZero 调用链

```
test_client_AR.py
  │ obs dict (OpenPI format: observation/xxx keys)
  ▼
socket_test_optimized_AR.py: ARDroidRoboarenaPolicy.infer(obs)
  │ 1. _convert_observation(obs)     ← key映射 + 帧累积 + reshape
  │ 2. dist.broadcast(signal)        ← 分布式通信
  │ 3. Batch(obs=converted_obs)      ← 包成 tianshou.Batch
  ▼
sim_policy.py: GrootSimPolicy.lazy_joint_forward_causal(batch)
  │ 1. unsqueeze_dict_values         ← 加 batch dim
  │ 2. apply(batch)                  ← 数据归一化 (action normalize等)
  │ 3. bf16 转换
  ▼
base_vla.py: VLA.lazy_joint_video_action_causal(normalized_input)
  │ 1. prepare_input(inputs)         ← 拆成 backbone_inputs + action_inputs
  │ 2. backbone(backbone_inputs)     ← backbone 是 Identity (不做事)
  ▼
wan_flow_matching_action_tf.py: WANPolicyHead.lazy_joint_video_action(backbone_out, action_input)
  │ 1. 视频预处理 (uint8→float, normalize, reshape)
  │ 2. text encoder (UMT5-XXL)
  │ 3. image encoder (CLIP) + VAE encode → clip_feas, ys
  │ 4. KV cache 创建/prefill
  │ 5. 去噪循环 (_run_diffusion_steps × N)
  │ 6. scheduler.step (video + action)
  ▼
wan_video_dit_action_casual_chunk.py: CausalWanModel.forward(...)
  │ (实际 DiT transformer forward)
  ▼
返回: BatchFeature(action_pred=..., video_pred=...)
  ▼
sim_policy.py: unapply(normalized_action)  ← 动作反归一化
  ▼
socket_test_optimized_AR.py: _convert_action(action_dict)  ← dict → ndarray(N,8)
  ▼
client 收到 ndarray(N, 8)
```

## vllm-omni 需要 vs 不需要

| 层 | DreamZero 原始 | vllm-omni 需要？ | 原因 |
|---|---|---|---|
| `tianshou.Batch` | 包装 obs | **不需要** | 纯 dict 直接传 |
| `GrootSimPolicy` | 模型加载+归一化+反归一化 | **部分需要** | 归一化/反归一化逻辑需要，但不要 tianshou/hydra 框架 |
| `base_vla.VLA` | prepare_input + backbone | **不需要** | backbone 是 Identity，prepare_input 只是拆 dict |
| `ActionHead` 基类 | 抽象接口 | **不需要** | 直接实现 |
| `dist.broadcast/barrier` | 分布式通信 | **不需要** | CFGParallelMixin 管 |
| `_convert_observation` | key 映射+帧累积 | **需要** | 放在 serving 层 |
| `apply/unapply` | 动作归一化/反归一化 | **需要** | 搬到 pipeline 层 |

## vllm-omni 目标调用链

```
test_client_AR.py (不改代码，只改 URL)
  │ obs dict (OpenPI format)
  ▼
openpi_connection.py: RobotRealtimeConnection.handle_connection()
  │ 1. _unpack(bytes) → obs dict
  │ 2. obs.pop("endpoint")
  ▼
openpi_serving.py: ServingRealtimeRobotOpenPI.infer(obs)
  │ 1. _build_request(obs) → OmniDiffusionRequest
  │    (key映射 + 帧累积在这里做)
  ▼
DiffusionEngine.step(request)
  ▼
pipeline_dreamzero.py: DreamZeroPipeline.forward(req)
  │ 1. 视频预处理 (uint8→float, normalize)
  │ 2. text encoder (复用 UMT5)
  │ 3. image encoder (CLIP) + VAE encode (复用 DistributedAutoencoderKLWan)
  │ 4. KV cache 创建/prefill
  │ 5. diffuse() 去噪循环
  │    ├─ predict_noise_maybe_with_cfg (CFGParallelMixin)
  │    ├─ scheduler_step_maybe_with_cfg (VideoActionScheduler)
  │    └─ _synchronize_cfg_parallel_step_output
  │ 6. 动作反归一化
  ▼
返回: DiffusionOutput(custom_output={"actions": ndarray(N,8)})
  ▼
openpi_serving.py: _extract_actions(result) → ndarray
  ▼
openpi_connection.py: _pack(actions) → send_bytes
  ▼
client 收到 ndarray(N, 8)
```

## 需要实现的文件

### 核心文件（需新写）

| 文件 | 对应 DreamZero 原始 | 说明 |
|------|-------------------|------|
| `pipeline_dreamzero.py` | `WANPolicyHead.lazy_joint_video_action` | 主 pipeline，去掉 tianshou/BatchFeature 包装 |
| `modeling/causal_wan_model.py` | `CausalWanModel` | 40层 DiT transformer |
| `modeling/wan_video_image_encoder.py` | `WanImageEncoder` | CLIP ViT-H/14 编码器 |

### 可复用 vllm-omni 现有组件

| 组件 | vllm-omni 现有 | 对应 DreamZero |
|------|--------------|---------------|
| Text Encoder | `UMT5EncoderModel` | `WanTextEncoder` |
| VAE | `DistributedAutoencoderKLWan` | `WanVideoVAE` |
| Scheduler | `FlowUniPCMultistepScheduler` | `FlowUniPCMultistepScheduler` |
| CFG Parallel | `CFGParallelMixin` | 手写 `dist.broadcast/isend/irecv` |

### 从 DreamZero 提取+精简

| 文件 | 对应 DreamZero 原始 | 说明 |
|------|-------------------|------|
| `modeling/action_encoder.py` | `MultiEmbodimentActionEncoder` 等 | 已完成 |
| `action_transform.py` | `GrootSimPolicy.apply/unapply` | 动作归一化/反归一化 |

## 关键精简点

1. **去掉 tianshou.Batch** — 纯 dict 传递，不需要 Batch 包装
2. **去掉 VLA/backbone** — backbone 是 Identity，直接跳过
3. **去掉 hydra/OmegaConf** — 不用 hydra 实例化，直接构造
4. **去掉 dist.broadcast/barrier** — CFGParallelMixin 的 all_gather 替代
5. **去掉 GrootSimPolicy** — 把 apply/unapply 逻辑直接搬到 pipeline
6. **去掉 ActionHead 基类** — 不需要抽象继承
