# stage_configs/ -- 多阶段流水线配置模块

## 文件概述

`stage_configs/` 目录包含所有支持模型的多阶段推理流水线 YAML 配置文件。每个 YAML 文件定义了一个完整的推理流水线，包括各阶段的模型架构、运行设备、引擎参数、采样参数以及阶段间的数据传输方式。

**目录路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_configs/`

## 模块结构

```
stage_configs/
├── __init__.py                          # 空文件
├── bagel.yaml                           # Bagel (AR + Diffusion)
├── bagel_multiconnector.yaml            # Bagel (分布式连接器版)
├── cosyvoice3.yaml                      # CosyVoice3 (Talker + Code2Wav)
├── fish_speech_s2_pro.yaml              # Fish Speech S2 Pro (Slow AR + DAC)
├── glm_image.yaml                       # GLM-Image (AR + Diffusion)
├── glm_image_muilticonnector.yaml       # GLM-Image (分布式连接器版)
├── hunyuan_image_3_moe.yaml             # Hunyuan-Image3 (单阶段 AR)
├── mammoth_moda2.yaml                   # MammothModa2 (AR + DiT)
├── mammoth_moda2_ar.yaml                # MammothModa2 (纯 AR 理解)
├── mimo_audio.yaml                      # MiMo-Audio (LLM + Code2Wav)
├── mimo_audio_async_chunk.yaml          # MiMo-Audio (异步分块流式)
├── qwen2_5_omni.yaml                    # Qwen2.5-Omni (3阶段)
├── qwen2_5_omni_multiconnector.yaml     # Qwen2.5-Omni (分布式版)
├── qwen3_omni_moe.yaml                  # Qwen3-Omni MoE (3阶段)
├── qwen3_omni_moe_async_chunk.yaml      # Qwen3-Omni MoE (异步分块)
├── qwen3_omni_moe_multiconnector.yaml   # Qwen3-Omni MoE (分布式版)
├── qwen3_tts.yaml                       # Qwen3-TTS (异步分块)
├── qwen3_tts_batch.yaml                 # Qwen3-TTS (批处理版)
└── qwen3_tts_no_async_chunk.yaml        # Qwen3-TTS (非异步版)
```

## 详细文档

所有 YAML 配置文件的详细解析请参见：[stage_configs_all.md](stage_configs_all.md)

## 与其他模块的关系

- **engine/**: 引擎启动时读取 YAML 配置初始化多阶段流水线
- **models/registry.py**: `model_arch` 字段映射到注册表中的模型架构
- **stage_input_processors/**: `custom_process_input_func` 和 `custom_process_next_stage_input_func` 字段引用处理器
- **core/sched/**: `scheduler_cls` 字段引用调度器类
- **worker/**: `worker_type` 或 `worker_cls` 字段指定 Worker 实现

## 总结

`stage_configs/` 是 vllm-omni 多阶段流水线的声明式配置中心，通过 YAML 文件定义了 10+ 种模型的完整推理流水线。配置覆盖同步/异步、单机/分布式等多种部署场景。
