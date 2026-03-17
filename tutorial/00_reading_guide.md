# vllm-omni 项目教程：阅读指南

> 基于提交 [`027a68f`](https://github.com/vllm-project/vllm-omni/commit/027a68f159e86000679001decfc76a4778a0079e)（`refactor` 分支，2026-03-16）

## 项目简介

vllm-omni 是一个多模态推理框架，支持文本、语音、图像、视频等多种模态的生成与处理。项目在 vLLM 的基础上，扩展了多阶段流水线编排、扩散模型推理、分布式多模态协调等能力，统一服务于 AR（自回归）模型和 Diffusion（扩散）模型。

---

## 项目整体架构

```
请求入口 (entrypoints/)
    │
    ▼
引擎层 (engine/)  ──── 请求编排、异步引擎、阶段协调
    │
    ├── 配置 (config/)  ──── 模型配置、阶段配置、YAML 解析
    │
    ├── 调度 (core/)  ──── 多种调度器实现
    │
    ├── 输入处理 (inputs/)  ──── 输入数据预处理
    │
    ├── GPU 执行 (worker/)  ──── GPU Worker、ModelRunner
    │
    ├── AR 模型 (model_executor/)  ──── Qwen3-Omni、Qwen3-TTS、Fish Speech、CosyVoice3 等
    │
    ├── 扩散模型 (diffusion/)  ──── Flux、SD3、Bagel、WAN、Helios 等
    │
    └── 分布式 (distributed/)  ──── OmniConnectors、OmniCoordinator、KV Transfer
```

---

## 模块概览

| 模块 | 路径 | 文件数 | 职责 |
|------|------|--------|------|
| **entrypoints** | `entrypoints/` | 38 | CLI 命令行工具与 OpenAI 兼容 API 服务 |
| **engine** | `engine/` | 9 | 请求编排、异步引擎、多阶段协调 |
| **config** | `config/` | 5 | 模型配置、阶段配置、LoRA 配置、YAML 工具 |
| **core** | `core/` | 5 | 调度器实现（多种调度策略） |
| **inputs** | `inputs/` | 3 | 输入数据处理与转换 |
| **worker** | `worker/` | 9 | GPU Worker 与 ModelRunner 实现 |
| **model_executor** | `model_executor/` | 93 | AR 模型实现（Qwen3-Omni、Fish Speech 等） |
| **diffusion** | `diffusion/` | 187 | 扩散模型引擎（attention、cache、模型、调度、量化等） |
| **distributed** | `distributed/` | 30 | 分布式推理（OmniConnectors、Coordinator、KV Transfer） |
| **platforms** | `platforms/` | 24 | 硬件平台适配（CUDA、ROCm、NPU、XPU） |
| **benchmarks** | `benchmarks/` | 7 | 性能基准测试工具 |
| **metrics** | `metrics/` | 3 | 监控指标采集 |
| **lora** | `lora/` | 3 | LoRA 适配器支持 |
| **tokenizers** | `tokenizers/` | 2 | 分词器封装 |
| **plugins** | `plugins/` | 1 | 插件机制 |
| **sample** | `sample/` | 1 | 采样逻辑 |
| **utils** | `utils/` | 1 | 通用工具函数 |
| **transformers_utils** | `transformers_utils/` | 4 | HuggingFace Transformers 集成工具 |

---

## 三条阅读路径

### 路径一：快速入门（约 2 小时）

适合希望快速了解整体工作流的读者。按以下顺序阅读核心模块即可掌握请求从进入到执行的完整链路。

```
1. entrypoints/  ──  了解请求如何进入系统
       │
2. engine/       ──  了解请求如何被编排和调度
       │
3. config/       ──  了解模型和阶段如何配置
       │
4. worker/       ──  了解模型如何在 GPU 上执行
```

**推荐阅读顺序：**

| 步骤 | 文件 | 说明 |
|------|------|------|
| 1 | `entrypoints/openai/` | OpenAI 兼容 API 的入口，理解请求格式 |
| 2 | `entrypoints/omni.py` | Omni 服务主入口 |
| 3 | `engine/async_omni_engine.py` | 异步引擎，请求生命周期管理 |
| 4 | `engine/orchestrator.py` | 多阶段编排器 |
| 5 | `config/model.py` | 模型配置定义 |
| 6 | `config/stage_config.py` | 阶段配置定义 |
| 7 | `worker/base.py` | Worker 基类 |
| 8 | `worker/gpu_ar_worker.py` | AR 模型 GPU Worker |

---

### 路径二：完整学习（按依赖顺序）

适合希望系统性掌握整个项目的读者。从底层基础设施向上层服务逐步推进。

```
阶段 1：基础设施
    config/  →  platforms/  →  utils/  →  plugins/

阶段 2：核心执行层
    inputs/  →  sample/  →  tokenizers/  →  worker/  →  core/

阶段 3：模型实现
    model_executor/  →  diffusion/

阶段 4：分布式与协调
    distributed/

阶段 5：引擎与服务
    engine/  →  entrypoints/

阶段 6：运维与生态
    metrics/  →  benchmarks/  →  lora/
```

**各阶段详细路线：**

**阶段 1 - 基础设施（约 2 小时）**
- `config/yaml_util.py` — YAML 解析工具
- `config/model.py` — 模型配置数据结构
- `config/stage_config.py` — 多阶段流水线配置
- `config/lora.py` — LoRA 配置
- `platforms/` — 硬件平台抽象层（先看 CUDA，再看其他平台）
- `utils/` — 通用工具
- `plugins/` — 插件加载机制

**阶段 2 - 核心执行层（约 3 小时）**
- `inputs/` — 输入数据预处理流程
- `sample/` — 采样策略实现
- `tokenizers/` — 分词器封装
- `worker/base.py` — Worker 基类定义
- `worker/gpu_model_runner.py` — 通用 GPU ModelRunner
- `worker/gpu_ar_model_runner.py` — AR 模型专用 Runner
- `worker/gpu_ar_worker.py` — AR 模型 Worker
- `worker/gpu_generation_model_runner.py` — 生成模型 Runner
- `worker/gpu_generation_worker.py` — 生成模型 Worker
- `worker/gpu_memory_utils.py` — GPU 显存管理
- `core/` — 调度器实现

**阶段 3 - 模型实现（约 6-8 小时）**
- `model_executor/registry.py` — 模型注册表
- `model_executor/layers/` — 公共层实现（注意力、旋转位置编码等）
- `model_executor/models/qwen3_omni/` — Qwen3-Omni 模型
- `model_executor/models/qwen3_tts/` — Qwen3-TTS 模型
- `model_executor/models/fish_speech/` — Fish Speech 模型
- `model_executor/models/cosyvoice3/` — CosyVoice3 模型
- `model_executor/models/mimo_audio/` — MiMo Audio 模型
- `model_executor/stage_configs/` — 各模型的阶段配置
- `diffusion/` — 扩散模型引擎（详见主题路径）

**阶段 4 - 分布式与协调（约 3 小时）**
- `distributed/omni_coordinator/` — 分布式协调器
- `distributed/omni_connectors/` — 多模态连接器
- `distributed/kv_transfer/` — KV Cache 传输
- `distributed/ray_utils/` — Ray 分布式工具

**阶段 5 - 引擎与服务（约 3 小时）**
- `engine/stage_init.py` — 阶段初始化
- `engine/stage_engine_core_client.py` — 阶段引擎客户端
- `engine/orchestrator.py` — 多阶段编排
- `engine/output_processor.py` — 输出处理
- `engine/async_omni_engine.py` — 异步引擎主逻辑
- `entrypoints/openai/` — OpenAI 兼容 API
- `entrypoints/cli/` — CLI 工具
- `entrypoints/omni.py` — Omni 服务入口

**阶段 6 - 运维与生态（约 1 小时）**
- `metrics/` — 监控指标
- `benchmarks/` — 性能测试
- `lora/` — LoRA 微调适配

---

### 路径三：按主题阅读

适合对特定领域感兴趣的读者，可直接跳到感兴趣的主题。

#### 主题 A：API 服务与请求处理

理解如何启动服务、处理 HTTP 请求、返回流式响应。

```
entrypoints/openai/         ── OpenAI 兼容 API 实现
entrypoints/cli/            ── 命令行工具
entrypoints/omni.py         ── Omni 主入口
entrypoints/chat_utils.py   ── 聊天工具函数
entrypoints/pd_utils.py     ── PD 分离部署工具
entrypoints/stage_utils.py  ── 阶段管理工具
```

#### 主题 B：引擎与调度

理解多阶段流水线如何编排和调度。

```
engine/orchestrator.py              ── 多阶段编排器
engine/async_omni_engine.py         ── 异步引擎
engine/stage_engine_core_client.py  ── 阶段引擎客户端
engine/output_processor.py          ── 输出后处理
core/                               ── 调度器实现
config/stage_config.py              ── 阶段配置
```

#### 主题 C：AR 模型（自回归模型）

理解语言模型和语音模型的推理实现。

```
model_executor/models/qwen3_omni/   ── Qwen3-Omni（多模态大模型）
model_executor/models/qwen3_tts/    ── Qwen3-TTS（语音合成）
model_executor/models/fish_speech/  ── Fish Speech（语音合成）
model_executor/models/cosyvoice3/   ── CosyVoice3（语音合成）
model_executor/models/mimo_audio/   ── MiMo Audio（音频处理）
model_executor/models/qwen2_5_omni/ ── Qwen2.5-Omni
model_executor/models/bagel/        ── Bagel 模型
model_executor/models/glm_image/    ── GLM 图像模型
model_executor/models/hunyuan_image3/ ── 混元图像模型
model_executor/models/mammoth_moda2/  ── Mammoth Moda2
model_executor/layers/              ── 公共层（注意力、位置编码等）
model_executor/model_loader/        ── 模型加载器
model_executor/stage_configs/       ── 各模型阶段配置
model_executor/stage_input_processors/ ── 阶段输入预处理
```

#### 主题 D：扩散模型

理解图像/视频/音频扩散模型的推理引擎。

```
diffusion/models/flux/          ── Flux 图像生成
diffusion/models/flux2/         ── Flux2
diffusion/models/flux2_klein/   ── Flux2 Klein
diffusion/models/sd3/           ── Stable Diffusion 3
diffusion/models/bagel/         ── Bagel 扩散模型
diffusion/models/wan2_2/        ── WAN 视频生成
diffusion/models/helios/        ── Helios
diffusion/models/ltx2/          ── LTX2 视频生成
diffusion/models/glm_image/     ── GLM 图像
diffusion/models/qwen_image/    ── Qwen 图像
diffusion/models/hunyuan_image_3/ ── 混元图像 3
diffusion/models/stable_audio/  ── Stable Audio
diffusion/models/cosyvoice3_audio/ ── CosyVoice3 音频扩散
diffusion/models/omnigen2/      ── OmniGen2
diffusion/models/ovis_image/    ── Ovis 图像
diffusion/models/schedulers/    ── 扩散调度器（噪声调度策略）
diffusion/attention/            ── 注意力机制（含 Ring Attention）
diffusion/cache/                ── 推理缓存（TeaCache 等）
diffusion/distributed/          ── 扩散模型分布式并行
diffusion/quantization/         ── 量化支持
diffusion/lora/                 ── 扩散模型 LoRA
diffusion/executor/             ── 扩散执行器
diffusion/worker/               ── 扩散 Worker
diffusion/model_loader/         ── 模型加载（含 GGUF 适配）
diffusion/layers/               ── 扩散公共层
diffusion/hooks/                ── 钩子机制
diffusion/offloader/            ── 显存卸载
diffusion/profiler/             ── 性能分析
diffusion/utils/                ── 扩散工具函数
```

#### 主题 E：分布式推理

理解多节点、多 GPU 协调推理机制。

```
distributed/omni_coordinator/   ── 分布式协调器（跨节点任务分配）
distributed/omni_connectors/    ── 多模态连接器（阶段间数据传输）
distributed/kv_transfer/        ── KV Cache 分布式传输
distributed/ray_utils/          ── Ray 集群管理工具
```

#### 主题 F：硬件平台适配

理解不同硬件平台上的兼容与优化。

```
platforms/cuda/     ── NVIDIA CUDA 平台
platforms/rocm/     ── AMD ROCm 平台
platforms/npu/      ── 华为昇腾 NPU 平台
platforms/xpu/      ── Intel XPU 平台
```

#### 主题 G：运维与工具

```
metrics/            ── Prometheus 监控指标
benchmarks/         ── 性能基准测试
lora/               ── LoRA 适配器管理
tokenizers/         ── 分词器封装
transformers_utils/ ── HuggingFace 集成工具
plugins/            ── 插件机制
```

---

## 完整文档索引

以下为 tutorial 目录下各模块文档的预期位置（使用相对路径链接）。

### 核心模块

| 文档 | 链接 |
|------|------|
| **本文 - 阅读指南** | [00_reading_guide.md](./00_reading_guide.md) |

### entrypoints - 服务入口

| 文档 | 链接 |
|------|------|
| entrypoints 概览 | [entrypoints/](./entrypoints/) |
| OpenAI 兼容 API | [entrypoints/openai/](./entrypoints/openai/) |
| OpenAI 协议定义 | [entrypoints/openai/protocol/](./entrypoints/openai/protocol/) |
| CLI 命令行工具 | [entrypoints/cli/](./entrypoints/cli/) |
| CLI 基准测试 | [entrypoints/cli/benchmark/](./entrypoints/cli/benchmark/) |

### engine - 引擎层

| 文档 | 链接 |
|------|------|
| engine 概览 | [engine/](./engine/) |

### config - 配置

| 文档 | 链接 |
|------|------|
| config 概览 | [config/](./config/) |

### core - 调度器

| 文档 | 链接 |
|------|------|
| core 概览 | [core/](./core/) |
| 调度器实现 | [core/sched/](./core/sched/) |

### inputs - 输入处理

| 文档 | 链接 |
|------|------|
| inputs 概览 | [inputs/](./inputs/) |

### worker - GPU 执行

| 文档 | 链接 |
|------|------|
| worker 概览 | [worker/](./worker/) |

### model_executor - AR 模型

| 文档 | 链接 |
|------|------|
| model_executor 概览 | [model_executor/](./model_executor/) |
| 公共层 | [model_executor/layers/](./model_executor/layers/) |
| 旋转位置编码 | [model_executor/layers/rotary_embedding/](./model_executor/layers/rotary_embedding/) |
| 模型加载器 | [model_executor/model_loader/](./model_executor/model_loader/) |
| 阶段配置 | [model_executor/stage_configs/](./model_executor/stage_configs/) |
| 阶段输入处理器 | [model_executor/stage_input_processors/](./model_executor/stage_input_processors/) |
| Qwen3-Omni | [model_executor/models/qwen3_omni/](./model_executor/models/qwen3_omni/) |
| Qwen3-TTS | [model_executor/models/qwen3_tts/](./model_executor/models/qwen3_tts/) |
| Qwen3-TTS 12Hz 分词器 | [model_executor/models/qwen3_tts/tokenizer_12hz/](./model_executor/models/qwen3_tts/tokenizer_12hz/) |
| Qwen3-TTS 25Hz 分词器 | [model_executor/models/qwen3_tts/tokenizer_25hz/](./model_executor/models/qwen3_tts/tokenizer_25hz/) |
| Qwen2.5-Omni | [model_executor/models/qwen2_5_omni/](./model_executor/models/qwen2_5_omni/) |
| Fish Speech | [model_executor/models/fish_speech/](./model_executor/models/fish_speech/) |
| CosyVoice3 | [model_executor/models/cosyvoice3/](./model_executor/models/cosyvoice3/) |
| MiMo Audio | [model_executor/models/mimo_audio/](./model_executor/models/mimo_audio/) |
| Bagel | [model_executor/models/bagel/](./model_executor/models/bagel/) |
| GLM Image | [model_executor/models/glm_image/](./model_executor/models/glm_image/) |
| Hunyuan Image3 | [model_executor/models/hunyuan_image3/](./model_executor/models/hunyuan_image3/) |
| Mammoth Moda2 | [model_executor/models/mammoth_moda2/](./model_executor/models/mammoth_moda2/) |

### diffusion - 扩散模型

| 文档 | 链接 |
|------|------|
| diffusion 概览 | [diffusion/](./diffusion/) |
| 注意力机制 | [diffusion/attention/](./diffusion/attention/) |
| 注意力后端 | [diffusion/attention/backends/](./diffusion/attention/backends/) |
| Ring Attention | [diffusion/attention/backends/ring/](./diffusion/attention/backends/ring/) |
| 注意力并行 | [diffusion/attention/parallel/](./diffusion/attention/parallel/) |
| 推理缓存 | [diffusion/cache/](./diffusion/cache/) |
| TeaCache | [diffusion/cache/teacache/](./diffusion/cache/teacache/) |
| 扩散分布式 | [diffusion/distributed/](./diffusion/distributed/) |
| 自编码器分布式 | [diffusion/distributed/autoencoders/](./diffusion/distributed/autoencoders/) |
| 扩散执行器 | [diffusion/executor/](./diffusion/executor/) |
| 钩子机制 | [diffusion/hooks/](./diffusion/hooks/) |
| 扩散公共层 | [diffusion/layers/](./diffusion/layers/) |
| 扩散 LoRA | [diffusion/lora/](./diffusion/lora/) |
| 扩散 LoRA 层 | [diffusion/lora/layers/](./diffusion/lora/layers/) |
| 扩散模型加载 | [diffusion/model_loader/](./diffusion/model_loader/) |
| GGUF 适配 | [diffusion/model_loader/gguf_adapters/](./diffusion/model_loader/gguf_adapters/) |
| 显存卸载 | [diffusion/offloader/](./diffusion/offloader/) |
| 性能分析 | [diffusion/profiler/](./diffusion/profiler/) |
| 量化支持 | [diffusion/quantization/](./diffusion/quantization/) |
| 扩散工具 | [diffusion/utils/](./diffusion/utils/) |
| 扩散 Worker | [diffusion/worker/](./diffusion/worker/) |
| 调度器 | [diffusion/models/schedulers/](./diffusion/models/schedulers/) |
| Flux | [diffusion/models/flux/](./diffusion/models/flux/) |
| Flux2 | [diffusion/models/flux2/](./diffusion/models/flux2/) |
| Flux2 Klein | [diffusion/models/flux2_klein/](./diffusion/models/flux2_klein/) |
| SD3 | [diffusion/models/sd3/](./diffusion/models/sd3/) |
| Bagel | [diffusion/models/bagel/](./diffusion/models/bagel/) |
| WAN 2.2 | [diffusion/models/wan2_2/](./diffusion/models/wan2_2/) |
| Helios | [diffusion/models/helios/](./diffusion/models/helios/) |
| LTX2 | [diffusion/models/ltx2/](./diffusion/models/ltx2/) |
| GLM Image | [diffusion/models/glm_image/](./diffusion/models/glm_image/) |
| Qwen Image | [diffusion/models/qwen_image/](./diffusion/models/qwen_image/) |
| Hunyuan Image 3 | [diffusion/models/hunyuan_image_3/](./diffusion/models/hunyuan_image_3/) |
| Stable Audio | [diffusion/models/stable_audio/](./diffusion/models/stable_audio/) |
| CosyVoice3 Audio | [diffusion/models/cosyvoice3_audio/](./diffusion/models/cosyvoice3_audio/) |
| OmniGen2 | [diffusion/models/omnigen2/](./diffusion/models/omnigen2/) |
| Ovis Image | [diffusion/models/ovis_image/](./diffusion/models/ovis_image/) |
| DreamID Omni | [diffusion/models/dreamid_omni/](./diffusion/models/dreamid_omni/) |
| Longcat Image | [diffusion/models/longcat_image/](./diffusion/models/longcat_image/) |
| Mammoth Moda2 | [diffusion/models/mammoth_moda2/](./diffusion/models/mammoth_moda2/) |
| NextStep 1.1 | [diffusion/models/nextstep_1_1/](./diffusion/models/nextstep_1_1/) |
| Z Image | [diffusion/models/z_image/](./diffusion/models/z_image/) |

### distributed - 分布式推理

| 文档 | 链接 |
|------|------|
| distributed 概览 | [distributed/](./distributed/) |
| OmniConnectors | [distributed/omni_connectors/](./distributed/omni_connectors/) |
| 连接器实现 | [distributed/omni_connectors/connectors/](./distributed/omni_connectors/connectors/) |
| Transfer Adapter | [distributed/omni_connectors/transfer_adapter/](./distributed/omni_connectors/transfer_adapter/) |
| 连接器工具 | [distributed/omni_connectors/utils/](./distributed/omni_connectors/utils/) |
| OmniCoordinator | [distributed/omni_coordinator/](./distributed/omni_coordinator/) |
| KV Transfer | [distributed/kv_transfer/](./distributed/kv_transfer/) |
| Ray 工具 | [distributed/ray_utils/](./distributed/ray_utils/) |

### platforms - 硬件平台

| 文档 | 链接 |
|------|------|
| platforms 概览 | [platforms/](./platforms/) |
| CUDA | [platforms/cuda/](./platforms/cuda/) |
| ROCm | [platforms/rocm/](./platforms/rocm/) |
| NPU | [platforms/npu/](./platforms/npu/) |
| XPU | [platforms/xpu/](./platforms/xpu/) |

### 辅助模块

| 文档 | 链接 |
|------|------|
| benchmarks 概览 | [benchmarks/](./benchmarks/) |
| metrics 概览 | [metrics/](./metrics/) |
| lora 概览 | [lora/](./lora/) |
| sample 概览 | [sample/](./sample/) |
| tokenizers 概览 | [tokenizers/](./tokenizers/) |
| plugins 概览 | [plugins/](./plugins/) |
| utils 概览 | [utils/](./utils/) |
| transformers_utils 概览 | [transformers_utils/](./transformers_utils/) |
| assets 资源 | [assets/](./assets/) |
| javascripts | [javascripts/](./javascripts/) |

---

## 阅读建议

1. **先看架构，再看实现**：先通过本指南理解整体架构和模块职责，再深入具体代码。
2. **以请求为线索**：推荐以一个完整请求（如文本生成或语音合成）为线索，从 entrypoints 开始跟踪到 worker 执行，串联理解各模块。
3. **配置先行**：阅读模型实现前，先理解 `config/stage_config.py` 中的多阶段流水线配置，这是理解模型编排的关键。
4. **AR 与 Diffusion 分开看**：这两类模型的执行路径差异较大，建议分别学习。AR 模型看 `model_executor/` + `worker/gpu_ar_*`，扩散模型看 `diffusion/` 自成体系。
5. **分布式最后看**：分布式模块依赖对单机执行流程的理解，建议掌握单机推理后再阅读。

---

## 核心概念速查

| 概念 | 说明 |
|------|------|
| **Stage（阶段）** | vllm-omni 将推理拆分为多个阶段（如 prefill、decode、vocoder），每个阶段可独立配置和执行 |
| **Orchestrator（编排器）** | 负责协调多个 Stage 的执行顺序和数据流转 |
| **OmniConnector（连接器）** | 在分布式场景下连接不同阶段、不同节点间的数据传输通道 |
| **OmniCoordinator（协调器）** | 管理分布式推理中的全局任务分配和状态同步 |
| **AR Model（自回归模型）** | 逐 token 生成的模型（如 LLM、TTS），使用 KV Cache |
| **Diffusion Model（扩散模型）** | 通过去噪过程生成图像/视频/音频的模型 |
| **ModelRunner** | 在 Worker 内部负责执行具体模型前向推理的组件 |
| **Pipeline YAML** | 定义多阶段流水线的配置文件，指定各阶段的模型、资源和连接关系 |
