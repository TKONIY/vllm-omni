# YAML 阶段配置文件 -- 全部配置详解

## 文件概述

本文档对 `stage_configs/` 目录下所有 YAML 配置文件进行统一解析。这些配置文件定义了 vllm-omni 中各模型的多阶段推理流水线，是系统运行的核心配置。

**目录路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_configs/`

## 配置文件通用结构

每个 YAML 配置文件包含以下顶级字段：

```yaml
async_chunk: bool          # 是否启用异步分块流式传输（可选）
stage_args:                # 阶段定义列表
  - stage_id: int          # 阶段 ID（从 0 开始）
    stage_type: str        # 阶段类型：llm / diffusion
    runtime:               # 运行时配置
      devices: str         # GPU 设备号
      max_batch_size: int  # 最大批处理大小
    engine_args:           # 引擎参数
      model_stage: str     # 模型阶段名（thinker/talker/code2wav/ar/dit 等）
      model_arch: str      # 模型架构名（对应注册表中的键）
      worker_type: str     # Worker 类型：ar / generation
      scheduler_cls: str   # 调度器类全路径
      # ... 更多引擎参数
    engine_input_source: [int]           # 输入来源阶段 ID（可选）
    custom_process_input_func: str       # 同步阶段间处理函数（可选）
    final_output: bool                   # 是否输出最终结果
    final_output_type: str               # 输出类型：text/audio/image
    default_sampling_params:             # 默认采样参数
      temperature: float
      top_p: float
      # ...

runtime:                   # 全局运行时配置
  enabled: bool
  defaults:
    window_size: int       # 触发窗口大小（-1 = 等待完成）
    max_inflight: int      # 最大并发数
  connectors:              # 连接器配置（可选）
    connector_name:
      name: str            # 连接器类名
      extra: {}            # 额外参数
  edges:                   # 阶段间连接边
    - from: int
      to: int
      window_size: int
```

## 关键配置字段详解

### engine_args 关键字段

| 字段 | 说明 | 示例值 |
|------|------|--------|
| `model_stage` | 模型阶段标识 | `thinker`, `talker`, `code2wav`, `ar`, `dit` |
| `model_arch` | 注册表中的架构名 | `Qwen3OmniMoeForConditionalGeneration` |
| `worker_type` | Worker 类型 | `ar`（自回归）, `generation`（生成） |
| `worker_cls` | 自定义 Worker 类 | `vllm_omni.worker.gpu_ar_worker.GPUARWorker` |
| `scheduler_cls` | 调度器类 | `vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler` |
| `engine_output_type` | 引擎输出类型 | `text`, `latent`, `audio`, `image`, `token_ids` |
| `hf_config_name` | HF 配置子名称 | `thinker_config`, `talker_config` |
| `model_subdir` | 模型配置子目录 | `vision_language_encoder` |
| `gpu_memory_utilization` | GPU 内存使用比例 | `0.3` ~ `0.9` |
| `tensor_parallel_size` | 张量并行大小 | `1`, `4`, `8` |
| `custom_process_next_stage_input_func` | 异步分块处理函数 | 完整的 Python 函数路径 |

### async_chunk 模式

当 `async_chunk: true` 时，阶段间通过流式分块传输数据：
- Stage 0 在每一步推理后调用 `custom_process_next_stage_input_func` 将中间结果发送给下游
- Stage 1 不必等待上游完成即可开始处理
- 通过 `connectors` 配置的 `codec_chunk_frames` 和 `codec_left_context_frames` 控制分块大小和重叠

### 连接器类型

| 连接器 | 说明 |
|--------|------|
| `SharedMemoryConnector` | 基于共享内存的本地传输 |
| `mooncake_connector` | 分布式场景的远程传输 |

---

## 各模型配置详解

### 1. Qwen2.5-Omni (`qwen2_5_omni.yaml`)

**流水线**: Thinker (Stage 0) -> Talker (Stage 1) -> Code2Wav (Stage 2)

**架构**: 3 阶段经典语音生成流水线

```
Stage 0 (Thinker): 多模态理解 + 文本生成
  - GPU: 0, model_arch: Qwen2_5OmniForConditionalGeneration
  - 输出: latent（隐藏状态传给 Talker）+ text（文本输出）

Stage 1 (Talker): 文本嵌入 -> 编解码
  - GPU: 1, 同一 model_arch
  - 输入处理: qwen2_5_omni.thinker2talker
  - 输出: latent（编码传给 Code2Wav）

Stage 2 (Code2Wav): 编码 -> 音频波形
  - GPU: 0, worker_type: generation
  - 输出: audio
```

**验证环境**: 2x H100-80G GPU

**变体**: `qwen2_5_omni_multiconnector.yaml` 使用 `mooncake_connector` 分布式连接器

---

### 2. Qwen3-Omni MoE (`qwen3_omni_moe.yaml`)

**流水线**: Thinker (Stage 0) -> Talker (Stage 1) -> Code2Wav (Stage 2)

**架构**: 3 阶段 MoE 语音生成流水线

```
Stage 0 (Thinker): 多模态 MoE 理解
  - GPU: 0, max_batch_size: 64
  - hf_config_name: thinker_config
  - 输出: latent

Stage 1 (Talker): 编码预测
  - GPU: 1, max_batch_size: 64
  - hf_config_name: talker_config
  - 输入处理: qwen3_omni.thinker2talker
  - stop_token_ids: [2150]

Stage 2 (Code2Wav): 波形生成
  - GPU: 1, worker_type: generation
  - 输入处理: qwen3_omni.talker2code2wav
  - 输出: audio
```

**变体**:
- `qwen3_omni_moe_async_chunk.yaml`: 启用异步分块（`async_chunk: true`），Stage 0 和 1 使用 `custom_process_next_stage_input_func` 和 `SharedMemoryConnector`
- `qwen3_omni_moe_multiconnector.yaml`: 使用 `mooncake_connector` 分布式连接器

---

### 3. Qwen3-TTS (`qwen3_tts.yaml`)

**流水线**: Talker (Stage 0) -> Code2Wav (Stage 1)

**架构**: 2 阶段纯 TTS 流水线，默认异步分块

```yaml
async_chunk: true
```

```
Stage 0 (Talker):
  - model_arch: Qwen3TTSTalkerForConditionalGeneration
  - hf_overrides.architectures: [Qwen3TTSTalkerForConditionalGeneration]
  - custom_process_next_stage_input_func: qwen3_tts.talker2code2wav_async_chunk
  - max_num_batched_tokens: 512
  - max_model_len: 4096

Stage 1 (Code2Wav):
  - model_arch: Qwen3TTSCode2Wav
  - worker_type: generation
  - max_num_batched_tokens: 8192
  - max_model_len: 32768
```

**连接器配置**:
```yaml
connectors:
  connector_of_shared_memory:
    extra:
      codec_streaming: true
      codec_chunk_frames: 25
      codec_left_context_frames: 25
```

**变体**:
- `qwen3_tts_no_async_chunk.yaml`: 非异步版（`async_chunk: false`），使用同步 `talker2code2wav`
- `qwen3_tts_batch.yaml`: 批处理版，更大的 batch_size

---

### 4. Fish Speech S2 Pro (`fish_speech_s2_pro.yaml`)

**流水线**: Slow AR (Stage 0) -> DAC Decoder (Stage 1)

**架构**: 2 阶段语音合成，异步分块

```
Stage 0 (Slow AR):
  - model_arch: FishSpeechSlowARForConditionalGeneration
  - custom_process_next_stage_input_func: fish_speech.slow_ar_to_dac_decoder_async_chunk
  - temperature: 0.8, top_k: 30
  - stop_token_ids: [151645] (<|im_end|>)

Stage 1 (DAC Decoder):
  - model_arch: FishSpeechDACDecoder
  - worker_type: generation
  - 输出: audio
```

---

### 5. CosyVoice3 (`cosyvoice3.yaml`)

**流水线**: Talker (Stage 0) -> Code2Wav (Stage 1)

**架构**: 2 阶段语音合成

```
Stage 0 (Talker):
  - model_arch: CosyVoice3Model
  - dtype: float32
  - disable_hybrid_kv_cache_manager: true

Stage 1 (Code2Wav):
  - 同一 model_arch，worker_cls: GPUGenerationWorker
  - 输入处理: cosyvoice3.text2flow
  - 输出: audio
```

特点：Stage 0 和 1 使用相同的 `model_arch` 但不同的 `model_stage`，模型内部根据 stage 加载不同的子模块。

---

### 6. MiMo-Audio (`mimo_audio.yaml`)

**流水线**: Fused Thinker+Talker (Stage 0) -> Code2Wav (Stage 1)

**架构**: 2 阶段，Stage 0 融合了 Thinker 和 Talker

```
Stage 0 (fused_thinker_talker):
  - model_arch: MiMoAudioForConditionalGeneration
  - dtype: bfloat16
  - max_model_len: 8192
  - 输出: text + latent

Stage 1 (Code2Wav):
  - 同一 model_arch, worker_type: generation
  - 输入处理: mimo_audio.llm2code2wav
  - 输出: audio
```

**变体**: `mimo_audio_async_chunk.yaml` 启用异步分块，使用 `mimo_audio.llm2code2wav_async_chunk`

**验证环境**: 1x H20-96G GPU

---

### 7. Bagel (`bagel.yaml`)

**流水线**: Thinker (Stage 0) -> Diffusion (Stage 1)

**架构**: 2 阶段图像生成，支持 CFG（Classifier-Free Guidance）

```
Stage 0 (Thinker):
  - model_arch: OmniBagelForConditionalGeneration
  - prompt_expand_func: bagel.expand_cfg_prompts  # CFG 提示扩展
  - max_batch_size: 3 (1 用户 + 2 CFG 伴随)
  - KV 缓存发送: need_send_cache: true
  - 输出: text

Stage 1 (Diffusion):
  - stage_type: diffusion
  - cfg_kv_collect_func: bagel.collect_cfg_kv_caches  # 收集 CFG KV 缓存
  - KV 缓存接收: need_recv_cache: true
  - 输出: image
```

特殊配置：`prompt_expand_func` 和 `cfg_kv_collect_func` 是 Bagel 独有的 CFG 多路推理配置。

**变体**: `bagel_multiconnector.yaml` 使用 `mooncake_connector`

---

### 8. GLM-Image (`glm_image.yaml`)

**流水线**: AR (Stage 0) -> Diffusion (Stage 1)

**架构**: 2 阶段图像生成

```
Stage 0 (AR):
  - model_arch: GlmImageForConditionalGeneration
  - model_subdir: vision_language_encoder  # 配置在子目录
  - tokenizer_subdir: processor
  - engine_output_type: token_ids
  - max_tokens: 1281 (256 小图 + 1024 大图 + 1 EOS)

Stage 1 (Diffusion):
  - stage_type: diffusion
  - model_arch: GlmImagePipeline
  - 输入处理: glm_image.ar2diffusion
  - num_inference_steps: 50, guidance_scale: 1.5
  - 输出: image
```

**变体**: `glm_image_muilticonnector.yaml` 使用 `mooncake_connector`

---

### 9. MammothModa2 (`mammoth_moda2.yaml`)

**流水线**: AR (Stage 0) -> DiT (Stage 1)

**架构**: 2 阶段多模态图像生成

```
Stage 0 (AR):
  - model_arch: MammothModa2ForConditionalGeneration
  - worker_cls: GPUARWorker
  - max_model_len: 8192
  - 输出: latent

Stage 1 (DiT):
  - 同一 model_arch，worker_cls: GPUGenerationWorker
  - 输入处理: mammoth_moda2.ar2dit
  - 输出: image
```

**变体**: `mammoth_moda2_ar.yaml` 仅 AR 阶段（纯文本/图像理解，无图像生成）

---

### 10. Hunyuan-Image3 (`hunyuan_image_3_moe.yaml`)

**流水线**: 单阶段 AR

**架构**: 单阶段大规模图像生成

```
Stage 0 (AR):
  - model_arch: HunyuanImage3ForCausalMM
  - tensor_parallel_size: 8
  - devices: "0,1,2,3,4,5,6,7"
  - hf_overrides.rope_parameters: mrope_section: [0, 32, 32]
  - 输出: text（理解模式）
```

**验证环境**: 8x L40S-48G GPU

---

## 配置模式对比

### 同步 vs 异步分块

| 特性 | 同步模式 | 异步分块模式 |
|------|----------|-------------|
| `async_chunk` | `false` 或未设置 | `true` |
| 阶段间函数 | `custom_process_input_func` | `custom_process_next_stage_input_func` |
| 数据传输 | 等待上游完成后批量传输 | 每步推理后流式传输 |
| 连接器 | 可选（默认直接传递） | 必须配置 |
| 首字延迟 | 较高（等待完整生成） | 较低（分块传输） |
| 适用场景 | 理解任务、小模型 | 语音流式输出 |

### 流水线架构统计

| 阶段数 | 配置文件 | 模型 |
|--------|----------|------|
| 1 阶段 | `hunyuan_image_3_moe.yaml`, `mammoth_moda2_ar.yaml` | Hunyuan-Image3, MammothModa2 (理解) |
| 2 阶段 | 多数 TTS/图像配置 | Bagel, GLM-Image, CosyVoice3, Fish Speech, MiMo-Audio, Qwen3-TTS, MammothModa2 |
| 3 阶段 | Omni 系列配置 | Qwen2.5-Omni, Qwen3-Omni MoE |

## 与其他模块的关系

- **models/registry.py**: `model_arch` 字段必须对应注册表中已注册的架构名
- **stage_input_processors/**: `custom_process_input_func` 和 `custom_process_next_stage_input_func` 引用此模块中的处理函数
- **core/sched/**: `scheduler_cls` 字段引用 `OmniARScheduler` 或 `OmniGenerationScheduler`
- **worker/**: `worker_type`（ar/generation）或 `worker_cls` 决定使用的 Worker 实现
- **engine/**: 引擎启动时解析 YAML 配置并初始化各阶段

## 总结

`stage_configs/` 中的 19 个 YAML 配置文件覆盖了 vllm-omni 支持的所有模型和部署模式。核心设计包括：

1. **声明式流水线定义**: 通过 YAML 配置完整描述多阶段推理流程
2. **灵活的阶段组合**: 从 1 阶段（纯理解）到 3 阶段（Thinker-Talker-Code2Wav）
3. **同步/异步双模式**: 支持批量处理和流式处理两种模式
4. **多种连接器**: 支持共享内存和分布式远程传输
5. **每模型多变体**: 同一模型可提供同步/异步/分布式等多种配置
