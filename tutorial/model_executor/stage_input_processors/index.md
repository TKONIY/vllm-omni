# stage_input_processors/ -- 阶段间数据转换处理器模块

## 文件概述

`stage_input_processors/` 模块提供了各模型特化的阶段间数据转换逻辑。当多阶段流水线中一个阶段完成推理后，其输出需要经过处理器转换为下一个阶段的输入格式。这些处理器分为同步模式（等待完整输出后批量转换）和异步分块模式（每步推理后流式转换）两类。

**目录路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/`

## 模块结构

```
stage_input_processors/
├── __init__.py            # 空文件
├── bagel.py               # Bagel CFG 提示扩展与 KV 缓存收集
├── chunk_size_utils.py    # 动态分块大小计算工具
├── cosyvoice3.py          # CosyVoice3 文本->语音流
├── fish_speech.py         # Fish Speech Slow AR -> DAC 解码
├── glm_image.py           # GLM-Image AR -> Diffusion
├── mammoth_moda2.py       # MammothModa2 AR -> DiT
├── mimo_audio.py          # MiMo-Audio LLM -> Code2Wav
├── qwen2_5_omni.py        # Qwen2.5-Omni Thinker -> Talker
├── qwen3_omni.py          # Qwen3-Omni Thinker -> Talker -> Code2Wav
└── qwen3_tts.py           # Qwen3-TTS Talker -> Code2Wav
```

## 子文件导航

| 文件 | 说明 | 文档链接 |
|------|------|----------|
| `bagel.py` | Bagel CFG 处理 | [bagel.md](bagel.md) |
| `chunk_size_utils.py` | 分块大小工具 | [chunk_size_utils.md](chunk_size_utils.md) |
| `cosyvoice3.py` | CosyVoice3 处理 | [cosyvoice3.md](cosyvoice3.md) |
| `fish_speech.py` | Fish Speech 处理 | [fish_speech.md](fish_speech.md) |
| `glm_image.py` | GLM-Image 处理 | [glm_image.md](glm_image.md) |
| `mammoth_moda2.py` | MammothModa2 处理 | [mammoth_moda2.md](mammoth_moda2.md) |
| `mimo_audio.py` | MiMo-Audio 处理 | [mimo_audio.md](mimo_audio.md) |
| `qwen2_5_omni.py` | Qwen2.5-Omni 处理 | [qwen2_5_omni.md](qwen2_5_omni.md) |
| `qwen3_omni.py` | Qwen3-Omni 处理 | [qwen3_omni.md](qwen3_omni.md) |
| `qwen3_tts.py` | Qwen3-TTS 处理 | [qwen3_tts.md](qwen3_tts.md) |

## 处理器类型对比

| 类型 | 函数签名 | 调用时机 | 返回值 |
|------|----------|----------|--------|
| 同步处理器 | `func(stage_list, engine_input_source, prompt, ...)` | 上游阶段完成后 | `list[OmniTokensPrompt]` |
| 异步分块处理器 | `func(transfer_manager, pooling_output, request, is_finished)` | 每步推理后 | `dict \| None` |

## 与其他模块的关系

- **stage_configs/**: YAML 配置中的 `custom_process_input_func` 和 `custom_process_next_stage_input_func` 引用此模块中的函数
- **inputs/data.py**: `OmniTokensPrompt` 是处理器输出的主要数据类型
- **engine/**: 引擎在阶段转换时调用相应的处理器函数
- **models/**: 处理器从模型输出的 `multimodal_output` 字典中提取数据

## 总结

`stage_input_processors/` 是 vllm-omni 多阶段流水线的数据桥梁，为 10 种模型提供了专门的阶段间数据转换逻辑，涵盖语音（Thinker->Talker->Code2Wav）、图像（AR->Diffusion/DiT）以及混合模态等场景，并同时支持同步批处理和异步流式两种工作模式。
