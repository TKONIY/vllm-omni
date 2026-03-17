# vLLM-Omni 代码教程

!!! info "版本信息"
    本教程基于 vLLM-Omni 仓库 [`refactor`](https://github.com/vllm-project/vllm-omni/tree/refactor) 分支，提交 [`027a68f`](https://github.com/vllm-project/vllm-omni/commit/027a68f159e86000679001decfc76a4778a0079e)（2026-03-16）。

    提交信息：`Fix Base voice clone: use actual codec encoder for exact ref_code_len`

欢迎阅读 vLLM-Omni 代码教程！本教程将逐文件地为你详细讲解 vLLM-Omni 的完整源代码。

## 项目简介

vLLM-Omni 是一个基于 vLLM 扩展的**多模态推理与服务框架**，支持文本、图像、视频、音频等多种模态的模型推理与生成。它通过多阶段流水线（Pipeline）架构，将大语言模型（LLM）的自回归解码与扩散模型（Diffusion）的图像/视频/音频生成统一在一个高性能服务系统中。

**核心特性：**

- 支持 30+ 种模型（Qwen3-Omni、Fish Speech、Flux、Stable Diffusion 3 等）
- 多阶段流水线架构（Thinker → Talker → Code2Wav / Diffusion）
- 分布式推理（张量并行、流水线并行、数据并行、序列并行）
- 多硬件平台支持（CUDA、ROCm、NPU、XPU）
- OpenAI 兼容的 API 服务

## 架构总览图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户层 (User Layer)                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Python API   │  │   CLI        │  │  OpenAI 兼容 API 服务     │  │
│  │  Omni /       │  │  vllm serve  │  │  /v1/chat/completions    │  │
│  │  AsyncOmni    │  │              │  │  /v1/audio/speech        │  │
│  │              │  │              │  │  /v1/images/generations   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
└─────────┼─────────────────┼─────────────────────┼──────────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      引擎层 (Engine Layer)                          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  AsyncOmniEngine                              │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │   │
│  │  │ Orchestrator │  │ OutputProcess │  │ StageEngineCore    │  │   │
│  │  │ (请求调度)    │  │ (输出处理)    │  │ Client (阶段通信)  │  │   │
│  │  └─────────────┘  └──────────────┘  └────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Stage 0:        │ │  Stage 1:    │ │  Stage 2:        │
│  Thinker (LLM)   │ │  Talker      │ │  Code2Wav /      │
│  自回归理解/推理  │→│  语音编码生成 │→│  Diffusion       │
│  (AR Worker)     │ │  (AR Worker) │ │  (Gen Worker)    │
└──────────────────┘ └──────────────┘ └──────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     分布式层 (Distributed Layer)                     │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────┐   │
│  │ OmniConnector │  │ OmniCoordinator  │  │ KV Transfer         │   │
│  │ (阶段间通信)   │  │ (负载均衡)       │  │ (KV缓存传输)        │   │
│  └──────────────┘  └──────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 主要工作流程图

以 Qwen3-Omni 语音对话为例，展示请求从输入到输出的完整流程：

```
步骤 1          步骤 2           步骤 3          步骤 4
用户输入  ──→  输入预处理  ──→  引擎调度   ──→  Stage 0: Thinker
(文本+音频)    (音频编码、      (Orchestrator    (Qwen3 MoE LLM
               图像处理)        路由请求)        理解并生成文本)
                                                      │
                                                      ▼ (hidden states + codec tokens)
步骤 7          步骤 6           步骤 5
输出音频  ←──  Stage 2:    ←──  Stage 1: Talker
(WAV格式)      Code2Wav         (MoE Talker
               (波形合成、       预测语音编码、
               HiFiGAN/         MTP加速)
               BigVGAN)
```

以 Flux 图像生成为例：

```
步骤 1          步骤 2           步骤 3          步骤 4
用户输入  ──→  文本编码   ──→  DiT 去噪   ──→  VAE 解码
(文本提示)     (CLIP/T5        (Flow Matching   (潜空间→像素
               双编码器)       多步采样)        空间)
                                                    │
                                                    ▼
                                              步骤 5
                                              输出图像
                                              (PNG/JPEG)
```

## 代码-流程映射表

| 步骤 | 功能 | 源代码文件 | 教程文档 |
|------|------|-----------|---------|
| 用户入口 | Python API | `entrypoints/omni.py` | [教程](entrypoints/omni.py.md) |
| 用户入口 | CLI 启动服务 | `entrypoints/cli/serve.py` | [教程](entrypoints/cli/serve.py.md) |
| 用户入口 | OpenAI API | `entrypoints/openai/api_server.py` | [教程](entrypoints/openai/api_server.py.md) |
| 输入处理 | 输入预处理 | `inputs/preprocess.py` | [教程](inputs/preprocess.py.md) |
| 引擎调度 | 异步引擎 | `engine/async_omni_engine.py` | [教程](engine/async_omni_engine.md) |
| 引擎调度 | 编排器 | `engine/orchestrator.py` | [教程](engine/orchestrator.md) |
| 调度器 | AR 调度 | `core/sched/omni_ar_scheduler.py` | [教程](core/sched/omni_ar_scheduler.py.md) |
| 调度器 | 生成调度 | `core/sched/omni_generation_scheduler.py` | [教程](core/sched/omni_generation_scheduler.py.md) |
| Worker | AR 执行 | `worker/gpu_ar_worker.py` | [教程](worker/gpu_ar_worker.md) |
| Worker | 生成执行 | `worker/gpu_generation_worker.py` | [教程](worker/gpu_generation_worker.md) |
| 模型 | Qwen3-Omni | `model_executor/models/qwen3_omni/` | [教程](model_executor/models/qwen3_omni/index.md) |
| 模型 | Qwen3-TTS | `model_executor/models/qwen3_tts/` | [教程](model_executor/models/qwen3_tts/index.md) |
| 模型 | Fish Speech | `model_executor/models/fish_speech/` | [教程](model_executor/models/fish_speech/index.md) |
| 扩散引擎 | Diffusion Engine | `diffusion/diffusion_engine.py` | [教程](diffusion/diffusion_engine.md) |
| 扩散模型 | Flux | `diffusion/models/flux/` | [教程](diffusion/models/flux/index.md) |
| 扩散模型 | Stable Diffusion 3 | `diffusion/models/sd3/` | [教程](diffusion/models/sd3/index.md) |
| 分布式 | 连接器 | `distributed/omni_connectors/` | [教程](distributed/omni_connectors/index.md) |
| 分布式 | 协调器 | `distributed/omni_coordinator/` | [教程](distributed/omni_coordinator/index.md) |
| 配置 | 模型配置 | `config/model.py` | [教程](config/model.py.md) |
| 配置 | 阶段配置 | `config/stage_config.py` | [教程](config/stage_config.py.md) |

## 模块索引

| 模块 | 文档数 | 说明 |
|------|--------|------|
| [背景知识](00_background_knowledge.md) | 1 | 技术领域基础知识 |
| [阅读指南](00_reading_guide.md) | 1 | 三条阅读路径推荐 |
| [根模块](./) | 6 | 包入口、日志、输出、补丁、请求、版本 |
| [config/](config/index.md) | 6 | 配置系统（模型、阶段、YAML） |
| [core/](core/index.md) | 5 | 调度器核心（AR、生成调度） |
| [inputs/](inputs/index.md) | 3 | 输入数据类型与预处理 |
| [assets/](assets/index.md) | 2 | 资源工具（音视频处理） |
| [engine/](engine/index.md) | 10 | 引擎核心（异步引擎、编排器、输出处理） |
| [entrypoints/](entrypoints/index.md) | 43 | 用户入口（API、CLI、OpenAI服务） |
| [worker/](worker/index.md) | 10 | GPU Worker（AR、生成、模型运行器） |
| [model_executor/](model_executor/index.md) | 98 | 模型执行器与模型实现 |
| [diffusion/](diffusion/index.md) | 220 | 扩散模型引擎与模型实现 |
| [distributed/](distributed/index.md) | 38 | 分布式推理（连接器、协调器、KV传输） |
| [platforms/](platforms/index.md) | 32 | 硬件平台支持（CUDA、ROCm、NPU、XPU） |
| [benchmarks/](benchmarks/index.md) | 8 | 基准测试工具 |
| [metrics/](metrics/index.md) | 4 | 指标收集与统计 |
| [lora/](lora/index.md) | 4 | LoRA 适配支持 |
| [plugins/](plugins/index.md) | 2 | 插件系统 |
| [sample/](sample/index.md) | 2 | 采样工具 |
| [tokenizers/](tokenizers/index.md) | 3 | 自定义分词器 |
| [transformers_utils/](transformers_utils/index.md) | 5 | HuggingFace 配置扩展 |
| [utils/](utils/index.md) | 2 | 通用工具 |

## 快速开始

- **零基础？** 从 [背景知识](00_background_knowledge.md) 开始
- **想快速了解？** 查看 [阅读指南](00_reading_guide.md) 中的「快速入门」路径
- **关注特定模型？** 直接进入 [model_executor/](model_executor/index.md) 或 [diffusion/](diffusion/index.md)
- **关注服务部署？** 查看 [entrypoints/](entrypoints/index.md) 和 [engine/](engine/index.md)
