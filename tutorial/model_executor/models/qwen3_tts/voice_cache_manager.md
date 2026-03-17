# `voice_cache_manager.py` — 声音克隆缓存管理

## 文件概述

本文件实现了 `VoiceCacheManager`，负责管理声音克隆的缓存功能。核心安全特性：仅使用 safetensors 格式（不使用 pickle/torch.load），防止远程代码执行（RCE）攻击。

## 关键代码解析

### 1. 缓存数据结构

```python
@dataclass
class VoiceClonePromptItem:
    ref_code: torch.Tensor | None     # 参考音频 codec codes
    ref_spk_embedding: torch.Tensor   # 说话人嵌入向量
    x_vector_only_mode: bool          # 是否仅使用 x-vector
    icl_mode: bool                    # 是否使用 in-context learning
    ref_text: str | None = None       # 参考文本
```

### 2. 安全保存

```python
def save_voice_cache(self, speaker, audio_file_path, prompt_items):
    tensors = {}
    metadata = {}
    for i, item in enumerate(prompt_items):
        tensors[f"item_{i}_ref_spk_embedding"] = item.ref_spk_embedding.detach().cpu()
        tensors[f"item_{i}_x_vector_only_mode"] = torch.tensor(int(item.x_vector_only_mode))
        if item.ref_text is not None:
            metadata[f"item_{i}_ref_text"] = item.ref_text
    save_file(tensors, str(cache_file_path), metadata=metadata)
```

### 3. 安全加载（路径限制）

```python
def load_cached_voice_prompt(self, speaker, device=None):
    cache_file_path = Path(speaker_info.get("cache_file", "")).resolve()
    base_dir = Path(self.speech_voice_samples_dir).resolve()
    # 路径安全检查：防止目录遍历攻击
    if not str(cache_file_path).startswith(str(base_dir)):
        logger.error("Illegal cache path outside base dir")
        return None
    if cache_file_path.suffix != ".safetensors":
        logger.error("Legacy or unsafe cache format rejected")
        return None
    with safe_open(cache_file_path, framework="pt", device="cpu") as f:
        ...
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `VoiceClonePromptItem` | 数据类 | 声音克隆提示数据 |
| `VoiceCacheManager` | 类 | 缓存管理器 |
| `save_voice_cache()` | 方法 | 安全保存缓存 |
| `load_cached_voice_prompt()` | 方法 | 安全加载缓存 |
| `get_speaker_audio_path()` | 方法 | 获取说话人音频路径 |
| `update_metadata_cache_info()` | 方法 | 更新元数据 |

## 与其他模块的关系

- **被引用**: `qwen3_tts.py` 使用缓存管理器
- **依赖**: `MetadataManager` 管理元数据 JSON

## 总结

`VoiceCacheManager` 在保证安全性的前提下提供高效的声音克隆缓存。三层安全保障：(1) 仅使用 safetensors 格式；(2) 路径限制在 `speech_voice_samples_dir` 内；(3) 拒绝非 `.safetensors` 后缀的文件。
