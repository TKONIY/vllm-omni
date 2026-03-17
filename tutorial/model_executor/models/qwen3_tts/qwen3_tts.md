# `qwen3_tts.py` — 统一入口模型

## 文件概述

本文件实现了 `Qwen3TTSModelForGeneration`，即 Qwen3-TTS 在 vLLM 中的统一入口。它封装了 HuggingFace 的 `Qwen3TTSForConditionalGeneration` 模型，并提供 vLLM 适配接口。支持三种任务类型：CustomVoice（声音克隆）、VoiceDesign（声音设计）和 Base（基础合成）。

## 关键代码解析

### 1. 模型初始化

```python
class Qwen3TTSModelForGeneration(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, **attn_kwargs)
        self.task_type = _normalize_task_type(model_path.split("-")[-1])
        self._enable_decoder_cudagraph()
```

从模型路径推断任务类型，并尝试启用 CUDA Graph 加速。

### 2. 任务类型规范化

```python
_TASK_TYPE_CANONICAL = {
    "customvoice": "CustomVoice",
    "voicedesign": "VoiceDesign",
    "base": "Base",
}
```

### 3. 音频输入类型

```python
AudioLike = (
    str           # wav 路径、URL、base64
    | np.ndarray  # 波形（需要 sr）
    | tuple[np.ndarray, int]  # (波形, 采样率)
)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3TTSModelForGeneration` | 类 | vLLM 统一入口 |
| `_normalize_task_type()` | 函数 | 任务类型规范化 |
| `_enable_decoder_cudagraph()` | 方法 | 启用 CUDA Graph |

## 与其他模块的关系

- **依赖**: `configuration_qwen3_tts.py` 的配置类
- **依赖**: `voice_cache_manager.py` 的缓存管理
- **依赖**: HF 模型 `Qwen3TTSForConditionalGeneration`

## 总结

该文件是 Qwen3-TTS 的 vLLM 适配层，负责模型加载、任务类型推断和 CUDA Graph 初始化。
