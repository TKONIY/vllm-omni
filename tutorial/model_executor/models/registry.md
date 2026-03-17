# `registry.py` -- 模型注册表 OmniModelRegistry

## 文件概述

`registry.py` 定义了 vllm-omni 的全局模型注册表 `OmniModelRegistry`，它将 vLLM 原生模型与 Omni 扩展模型合并到一个统一的注册表中。引擎在加载模型时，通过模型架构名（如 `"Qwen3OmniMoeForConditionalGeneration"`）从注册表查找对应的模型类。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/models/registry.py`

## 关键代码解析

### Omni 模型注册字典 `_OMNI_MODELS`

```python
_OMNI_MODELS = {
    # 架构名: (模型子目录, 模块文件名, 类名)
    "Qwen3OmniMoeForConditionalGeneration": (
        "qwen3_omni", "qwen3_omni", "Qwen3OmniMoeForConditionalGeneration",
    ),
    "Qwen3TTSTalkerForConditionalGeneration": (
        "qwen3_tts", "qwen3_tts_talker", "Qwen3TTSTalkerForConditionalGeneration",
    ),
    "FishSpeechSlowARForConditionalGeneration": (
        "fish_speech", "fish_speech_slow_ar", "FishSpeechSlowARForConditionalGeneration",
    ),
    # ... 共 25+ 个模型架构注册
}
```

字典的三元组 `(mod_folder, mod_relname, cls_name)` 表示：
- `mod_folder`: 模型子目录名（如 `"qwen3_omni"`）
- `mod_relname`: 模块文件名（如 `"qwen3_omni_moe_thinker"`）
- `cls_name`: Python 类名（如 `"Qwen3OmniMoeThinkerForConditionalGeneration"`）

### 合并注册表

```python
_VLLM_OMNI_MODELS = {
    **_VLLM_MODELS,     # vLLM 原生模型
    **_OMNI_MODELS,     # Omni 扩展模型
}
```

### OmniModelRegistry 构建

```python
OmniModelRegistry = _ModelRegistry(
    {
        # vLLM 原生模型：使用 vllm.model_executor.models.xxx 路径
        **{
            model_arch: _LazyRegisteredModel(
                module_name=f"vllm.model_executor.models.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
        },
        # Omni 扩展模型：使用 vllm_omni.model_executor.models.xxx 路径
        **{
            model_arch: _LazyRegisteredModel(
                module_name=f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items()
        },
    }
)
```

关键区别在于模块路径：
- vLLM 原生模型使用 `vllm.model_executor.models.{模块名}` 路径
- Omni 模型使用 `vllm_omni.model_executor.models.{子目录}.{模块名}` 路径（多了一级子目录）

### 懒加载机制

`_LazyRegisteredModel` 支持懒加载：模型类不会在注册时导入，而是在首次使用时通过 `importlib` 动态加载。这避免了启动时导入所有模型的开销。

## 已注册模型一览

| 架构名 | 模型系列 | 用途 |
|--------|----------|------|
| `Qwen2_5OmniForConditionalGeneration` | Qwen2.5-Omni | 全模态入口 |
| `Qwen2_5OmniThinkerModel` | Qwen2.5-Omni | Thinker 阶段 |
| `Qwen2_5OmniTalkerModel` | Qwen2.5-Omni | Talker 阶段 |
| `Qwen2_5OmniToken2WavModel` | Qwen2.5-Omni | Token 到波形 |
| `Qwen3OmniMoeForConditionalGeneration` | Qwen3-Omni | MoE 全模态入口 |
| `Qwen3OmniMoeThinkerForConditionalGeneration` | Qwen3-Omni | MoE Thinker |
| `Qwen3OmniMoeTalkerForConditionalGeneration` | Qwen3-Omni | MoE Talker |
| `Qwen3OmniMoeCode2Wav` | Qwen3-Omni | Code 到波形 |
| `Qwen3TTSTalkerForConditionalGeneration` | Qwen3-TTS | TTS Talker |
| `Qwen3TTSCode2Wav` | Qwen3-TTS | TTS Code2Wav |
| `CosyVoice3Model` | CosyVoice3 | 语音合成 |
| `MammothModa2ForConditionalGeneration` | MammothModa2 | 多模态生成 |
| `MammothModa2DiTPipeline` | MammothModa2 | DiT 图像生成 |
| `MiMoAudioModel` | MiMo-Audio | 语音对话入口 |
| `MiMoAudioLLMModel` | MiMo-Audio | LLM 阶段 |
| `MiMoAudioToken2WavModel` | MiMo-Audio | Token 到波形 |
| `GlmImageForConditionalGeneration` | GLM-Image | 图像 AR 生成 |
| `OmniBagelForConditionalGeneration` | Bagel | CFG 图像生成 |
| `HunyuanImage3ForCausalMM` | Hunyuan-Image3 | 图像生成 |
| `FishSpeechSlowARForConditionalGeneration` | Fish Speech | Slow AR 语音 |
| `FishSpeechDACDecoder` | Fish Speech | DAC 解码器 |

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_OMNI_MODELS` | 字典 | Omni 扩展模型的注册配置 |
| `_VLLM_OMNI_MODELS` | 字典 | 合并后的完整模型配置 |
| `OmniModelRegistry` | `_ModelRegistry` 实例 | 全局模型注册表 |

## 与其他模块的关系

- **vllm.model_executor.models.registry**: 提供 `_VLLM_MODELS`、`_LazyRegisteredModel`、`_ModelRegistry` 基础设施
- **engine/**: 引擎通过 `OmniModelRegistry` 根据 YAML 配置中的 `model_arch` 查找模型类
- **stage_configs/**: YAML 中的 `model_arch` 字段必须是注册表中存在的键
- **models/ 各子目录**: 注册表指向这些子目录中的具体模型实现

## 总结

`registry.py` 是 vllm-omni 模型发现机制的核心，它通过合并 vLLM 原生注册表和 Omni 扩展注册表，实现了对 25+ 种模型架构的统一管理和懒加载。YAML 配置中的 `model_arch` 字段直接映射到此注册表中的键，使得添加新模型只需在 `_OMNI_MODELS` 字典中添加一行配置即可。
