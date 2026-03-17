# vLLM-Omni 教程文档追踪清单

> 本文件追踪所有需要编写教程文档的 Python 源文件。每个 `.py` 文件对应一份 `.md` 教程文档。
>
> **项目**: vllm-omni — 多模态推理框架
>
> **总计**: 约 466 个 Python 源文件

---

## 进度总览

| 模块 | 文件数 | 已完成 | 进度 |
|------|--------|--------|------|
| 根目录文件 | 6 | 0 | 0% |
| assets | 1 | 0 | 0% |
| config | 5 | 0 | 0% |
| core/sched | 5 | 0 | 0% |
| inputs | 3 | 0 | 0% |
| benchmarks | 7 | 0 | 0% |
| metrics | 3 | 0 | 0% |
| lora | 3 | 0 | 0% |
| plugins | 1 | 0 | 0% |
| sample | 1 | 0 | 0% |
| tokenizers | 2 | 0 | 0% |
| transformers_utils | 4 | 0 | 0% |
| utils | 1 | 0 | 0% |
| worker | 9 | 0 | 0% |
| engine | 9 | 0 | 0% |
| entrypoints | 38 | 0 | 0% |
| distributed | 30 | 0 | 0% |
| platforms | 24(+8 yaml) | 0 | 0% |
| model_executor | ~93(+20 yaml) | 0 | 0% |
| diffusion | ~180 | 0 | 0% |
| **合计** | **~466** | **0** | **0%** |

---

## 根目录文件（6 个）

> 框架顶层模块：初始化、日志、输出定义、补丁机制、请求封装、版本信息。

- [ ] `__init__.py`
- [ ] `logger.py`
- [ ] `outputs.py`
- [ ] `patch.py`
- [ ] `request.py`
- [ ] `version.py`

---

## assets — 资源处理（1 个）

> 视频等多媒体资源的加载与预处理。

- [ ] `video.py`

---

## config — 配置管理（5 个）

> 模型配置、LoRA 配置、多阶段 pipeline 配置及 YAML 解析工具。

- [ ] `__init__.py`
- [ ] `lora.py`
- [ ] `model.py`
- [ ] `stage_config.py`
- [ ] `yaml_util.py`

---

## core/sched — 核心调度器（5 个）

> 多模态推理的核心调度逻辑，包括自回归调度器和生成调度器。

- [ ] `core/__init__.py`
- [ ] `core/sched/__init__.py`
- [ ] `core/sched/omni_ar_scheduler.py`
- [ ] `core/sched/omni_generation_scheduler.py`
- [ ] `core/sched/output.py`

---

## inputs — 输入处理（3 个）

> 多模态输入数据的定义、封装与预处理。

- [ ] `__init__.py`
- [ ] `data.py`
- [ ] `preprocess.py`

---

## benchmarks — 基准测试（7 个）

> 性能基准测试：服务端压测、随机多模态数据集生成、指标采集与补丁。

- [ ] `serve.py`
- [ ] `data_modules/__init__.py`
- [ ] `data_modules/random_multi_modal_dataset.py`
- [ ] `metrics/__init__.py`
- [ ] `metrics/metrics.py`
- [ ] `patch/__init__.py`
- [ ] `patch/patch.py`

---

## metrics — 指标与监控（3 个）

> 推理过程中的统计指标定义与工具函数。

- [ ] `__init__.py`
- [ ] `stats.py`
- [ ] `utils.py`

---

## lora — LoRA 适配（3 个）

> LoRA 低秩适配的请求封装与工具函数。

- [ ] `__init__.py`
- [ ] `request.py`
- [ ] `utils.py`

---

## plugins — 插件系统（1 个）

> 插件注册与加载机制。

- [ ] `__init__.py`

---

## sample — 采样逻辑（1 个）

> 推理采样策略。

- [ ] `__init__.py`

---

## tokenizers — 分词器（2 个）

> 自定义分词器实现，包括 Mammoth MoDA2 专用分词器。

- [ ] `__init__.py`
- [ ] `mammoth_moda2_tokenizer.py`

---

## transformers_utils — Transformers 工具（4 个）

> HuggingFace Transformers 兼容层：自定义模型配置注册。

- [ ] `__init__.py`
- [ ] `configs/__init__.py`
- [ ] `configs/fish_speech.py`
- [ ] `configs/mammoth_moda2.py`

---

## utils — 通用工具（1 个）

> 跨模块共享的工具函数。

- [ ] `__init__.py`

---

## worker — 工作进程（9 个）

> GPU 工作进程：模型加载、推理执行、显存管理，涵盖自回归与生成两种模式。

- [ ] `__init__.py`
- [ ] `base.py`
- [ ] `gpu_ar_model_runner.py`
- [ ] `gpu_ar_worker.py`
- [ ] `gpu_generation_model_runner.py`
- [ ] `gpu_generation_worker.py`
- [ ] `gpu_memory_utils.py`
- [ ] `gpu_model_runner.py`
- [ ] `mixins.py`

---

## engine — 推理引擎（9 个）

> 异步推理引擎核心：参数解析、编排、输出处理、序列化、多阶段引擎客户端。

- [ ] `__init__.py`
- [ ] `arg_utils.py`
- [ ] `async_omni_engine.py`
- [ ] `orchestrator.py`
- [ ] `output_processor.py`
- [ ] `serialization.py`
- [ ] `stage_engine_core_client.py`
- [ ] `stage_init.py`
- [ ] `worker_cls_utils.py`

---

## entrypoints — 入口与 API（38 个）

> 服务入口：异步引擎、OpenAI 兼容 API、CLI 工具、聊天/语音/图像/视频接口。

### 顶层入口

- [ ] `__init__.py`
- [ ] `async_omni.py`
- [ ] `async_omni_diffusion.py`
- [ ] `cfg_companion_tracker.py`
- [ ] `chat_utils.py`
- [ ] `client_request_state.py`
- [ ] `omni_base.py`
- [ ] `omni.py`
- [ ] `pd_utils.py`
- [ ] `stage_utils.py`
- [ ] `utils.py`

### CLI 命令行工具

- [ ] `cli/__init__.py`
- [ ] `cli/logo.py`
- [ ] `cli/main.py`
- [ ] `cli/serve.py`

### CLI 基准测试

- [ ] `cli/benchmark/__init__.py`
- [ ] `cli/benchmark/base.py`
- [ ] `cli/benchmark/main.py`
- [ ] `cli/benchmark/serve.py`

### OpenAI 兼容 API 层

- [ ] `openai/__init__.py`
- [ ] `openai/api_server.py`
- [ ] `openai/audio_utils_mixin.py`
- [ ] `openai/errors.py`
- [ ] `openai/image_api_utils.py`
- [ ] `openai/metadata_manager.py`
- [ ] `openai/serving_chat.py`
- [ ] `openai/serving_speech.py`
- [ ] `openai/serving_speech_stream.py`
- [ ] `openai/serving_video.py`
- [ ] `openai/storage.py`
- [ ] `openai/stores.py`
- [ ] `openai/text_splitter.py`
- [ ] `openai/video_api_utils.py`

### OpenAI 协议定义

- [ ] `openai/protocol/__init__.py`
- [ ] `openai/protocol/audio.py`
- [ ] `openai/protocol/chat_completion.py`
- [ ] `openai/protocol/images.py`
- [ ] `openai/protocol/videos.py`

---

## distributed — 分布式通信（30 个）

> 分布式 KV 缓存传输、多节点连接器、协调器、Ray 工具。

### KV Transfer

- [ ] `kv_transfer/__init__.py`
- [ ] `kv_transfer/monkey_patch.py`

### Omni Connectors — 连接器框架

- [ ] `omni_connectors/__init__.py`
- [ ] `omni_connectors/adapter.py`
- [ ] `omni_connectors/factory.py`
- [ ] `omni_connectors/kv_transfer_manager.py`

### Omni Connectors — 连接器实现

- [ ] `omni_connectors/connectors/__init__.py`
- [ ] `omni_connectors/connectors/base.py`
- [ ] `omni_connectors/connectors/mooncake_store_connector.py`
- [ ] `omni_connectors/connectors/mooncake_transfer_engine_connector.py`
- [ ] `omni_connectors/connectors/shm_connector.py`
- [ ] `omni_connectors/connectors/yuanrong_connector.py`

### Omni Connectors — Transfer Adapter

- [ ] `omni_connectors/transfer_adapter/__init__.py`
- [ ] `omni_connectors/transfer_adapter/base.py`
- [ ] `omni_connectors/transfer_adapter/chunk_transfer_adapter.py`

### Omni Connectors — 工具

- [ ] `omni_connectors/utils/__init__.py`
- [ ] `omni_connectors/utils/config.py`
- [ ] `omni_connectors/utils/initialization.py`
- [ ] `omni_connectors/utils/kv_utils.py`
- [ ] `omni_connectors/utils/logging.py`
- [ ] `omni_connectors/utils/serialization.py`

### Omni Coordinator — 协调器

- [ ] `omni_coordinator/__init__.py`
- [ ] `omni_coordinator/load_balancer.py`
- [ ] `omni_coordinator/messages.py`
- [ ] `omni_coordinator/omni_coord_client_for_hub.py`
- [ ] `omni_coordinator/omni_coord_client_for_stage.py`
- [ ] `omni_coordinator/omni_coordinator.py`

### Ray 工具

- [ ] `ray_utils/__init__.py`
- [ ] `ray_utils/utils.py`

---

## platforms — 多平台适配（24 py + 8 yaml）

> 多硬件平台支持：CUDA、ROCm、NPU（昇腾）、XPU（Intel），含平台特定 worker 和模型实现。

### 平台接口

- [ ] `__init__.py`
- [ ] `interface.py`

### CUDA 平台

- [ ] `cuda/__init__.py`
- [ ] `cuda/platform.py`

### ROCm 平台

- [ ] `rocm/__init__.py`
- [ ] `rocm/platform.py`

### NPU 平台（昇腾）

- [ ] `npu/__init__.py`
- [ ] `npu/platform.py`
- [ ] `npu/models/__init__.py`
- [ ] `npu/models/hunyuan_fused_moe.py`
- [ ] `npu/worker/__init__.py`
- [ ] `npu/worker/npu_ar_model_runner.py`
- [ ] `npu/worker/npu_ar_worker.py`
- [ ] `npu/worker/npu_generation_model_runner.py`
- [ ] `npu/worker/npu_generation_worker.py`
- [ ] `npu/worker/npu_model_runner.py`

### XPU 平台（Intel）

- [ ] `xpu/__init__.py`
- [ ] `xpu/platform.py`
- [ ] `xpu/utils.py`
- [ ] `xpu/worker/__init__.py`
- [ ] `xpu/worker/xpu_ar_model_runner.py`
- [ ] `xpu/worker/xpu_ar_worker.py`
- [ ] `xpu/worker/xpu_generation_model_runner.py`
- [ ] `xpu/worker/xpu_generation_worker.py`

### YAML 阶段配置

- [ ] 各平台 `stage_configs/*.yaml` 文件（8 个）

---

## model_executor — 模型执行器（约 93 py + 20 yaml）

> 模型加载、模型层定义、阶段输入处理器，以及所有支持的模型实现。

### 模型加载器

- [ ] `model_loader/__init__.py`
- [ ] `model_loader/loader.py`
- [ ] `model_loader/utils.py`
- [ ] `model_loader/weight_utils.py`

### 模型层（layers/）

- [ ] `layers/__init__.py`
- [ ] `layers/activation.py`
- [ ] `layers/fused_moe/`（所有子文件）
- [ ] `layers/layernorm.py`
- [ ] `layers/linear.py`
- [ ] `layers/logits_processor.py`
- [ ] `layers/quantization/`（所有子文件）
- [ ] `layers/rotary_embedding.py`
- [ ] `layers/sampler.py`
- [ ] `layers/vocab_parallel_embedding.py`

### 阶段输入处理器

- [ ] `stage_input_processors/__init__.py`
- [ ] `stage_input_processors/` 各模型处理器

### 阶段配置（YAML）

- [ ] `stage_configs/` 各模型 YAML 配置（约 20 个）

### 模型实现（models/）

- [ ] `models/__init__.py`
- [ ] `models/chatglm.py`
- [ ] `models/clip.py`
- [ ] `models/cosyvoice2.py`
- [ ] `models/deepseek_v2.py`
- [ ] `models/fish_speech.py`
- [ ] `models/glm4voice.py`
- [ ] `models/hunyuan_dit.py`
- [ ] `models/internvl2.py`
- [ ] `models/llama.py`
- [ ] `models/mammoth_moda2.py`
- [ ] `models/minicpm.py`
- [ ] `models/minicpmo.py`
- [ ] `models/minimax_speech01.py`
- [ ] `models/qwen2.py`
- [ ] `models/qwen2_audio.py`
- [ ] `models/qwen2_vl.py`
- [ ] `models/qwen3_tts.py`
- [ ] `models/siglip.py`
- [ ] `models/spark_tts.py`
- [ ] `models/step_audio.py`
- [ ] `models/step_audio_tokenizer.py`
- [ ] `models/utils.py`
- [ ] `models/vita.py`
- [ ] `models/wan.py`
- [ ] 其他模型文件（以实际目录为准）

---

## diffusion — 扩散模型（约 180 py）

> 项目中最大的模块。涵盖扩散模型的完整推理流水线：注意力机制、缓存管理、分布式、执行器、钩子、LoRA、模型加载、量化、性能分析等。

### 注意力机制（attention/）

- [ ] `attention/__init__.py`
- [ ] `attention/backends/` 各后端实现
- [ ] `attention/layer.py`
- [ ] `attention/selector.py`

### 缓存管理（cache/）

- [ ] `cache/__init__.py`
- [ ] `cache/` 各缓存策略实现

### 分布式（distributed/）

- [ ] `distributed/__init__.py`
- [ ] `distributed/` 通信与并行策略

### 执行器（executor/）

- [ ] `executor/__init__.py`
- [ ] `executor/` 各执行器实现

### 钩子（hooks/）

- [ ] `hooks/__init__.py`
- [ ] `hooks/` 各钩子实现

### 模型层（layers/）

- [ ] `layers/__init__.py`
- [ ] `layers/` 扩散模型专用层

### LoRA（lora/）

- [ ] `lora/__init__.py`
- [ ] `lora/` 扩散模型 LoRA 适配

### 模型加载（model_loader/）

- [ ] `model_loader/__init__.py`
- [ ] `model_loader/` 扩散模型加载器

### 扩散模型实现（models/）

> 包含 20+ 子模型目录，每个对应一种扩散模型架构。

- [ ] `models/__init__.py`
- [ ] `models/cogvideox/`（CogVideoX 系列）
- [ ] `models/flux/`（Flux 系列）
- [ ] `models/hunyuan_video/`（混元视频）
- [ ] `models/kolors/`（可图）
- [ ] `models/mochi/`（Mochi）
- [ ] `models/sd/`（Stable Diffusion）
- [ ] `models/sdxl/`（SDXL）
- [ ] `models/sana/`（Sana）
- [ ] `models/step_video/`（Step Video）
- [ ] `models/wan/`（Wan）
- [ ] 其他子模型目录（以实际目录为准）

### 卸载器（offloader/）

- [ ] `offloader/__init__.py`
- [ ] `offloader/` 显存卸载策略

### 性能分析（profiler/）

- [ ] `profiler/__init__.py`
- [ ] `profiler/` 性能分析工具

### 量化（quantization/）

- [ ] `quantization/__init__.py`
- [ ] `quantization/` 各量化方案

### 工具（utils/）

- [ ] `utils/__init__.py`
- [ ] `utils/` 扩散模块工具函数

### 工作进程（worker/）

- [ ] `worker/__init__.py`
- [ ] `worker/` 扩散模型 worker 实现

---

## 编写规范

1. 每份教程文档应包含：模块概述、核心类/函数说明、使用示例、与其他模块的关系
2. `__init__.py` 的文档应着重描述模块的整体设计与公开接口
3. 完成一个文件的文档后，将对应的 `[ ]` 改为 `[x]` 并更新进度总览表
4. 优先级建议：engine > entrypoints > worker > config > distributed > model_executor > diffusion > 其他

---

*最后更新：2026-03-17*
