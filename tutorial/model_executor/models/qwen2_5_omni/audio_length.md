# `audio_length.py` — 音频长度对齐工具

## 文件概述

本文件提供纯 Python 实现的音频长度计算和对齐工具函数，用于 Token2Wav（codec → mel → wav）流程中确保梅尔频谱帧数与 codec token 数量保持正确的倍数关系。这在应用最大梅尔帧数限制时尤其重要。

## 关键代码解析

### 1. `resolve_max_mel_frames`

```python
def resolve_max_mel_frames(max_mel_frames: int | None, *, default: int = 30000) -> int:
    if max_mel_frames is not None:
        return int(max_mel_frames)
    return int(default)
```

简单的参数解析函数，提供默认值 30000 帧（约 10 分钟音频）。

### 2. `cap_and_align_mel_length`

```python
def cap_and_align_mel_length(
    *, code_len: int, repeats: int, max_mel_frames: int | None,
    default_max_mel_frames: int = 30000,
) -> tuple[int, int]:
```

核心函数，负责计算对齐后的 `(target_code_len, target_mel_len)` 对：

- `mel_len` 始终是 `repeats`（codec 扩展因子）的倍数
- 如果 `max_mel_frames <= 0`，不进行截断
- 确保至少保留一个 codec token 的梅尔帧数

**对齐逻辑**：
```python
target_duration = (target_duration // repeats) * repeats  # 向下对齐到 repeats 的倍数
target_code_len = target_duration // repeats              # 反算 codec 长度
```

## 核心类/函数

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `resolve_max_mel_frames` | `max_mel_frames`, `default` | `int` | 解析最大梅尔帧数 |
| `cap_and_align_mel_length` | `code_len`, `repeats`, `max_mel_frames` | `(int, int)` | 计算对齐后的 codec/mel 长度 |

## 与其他模块的关系

- 被 `qwen2_5_omni_token2wav.py` 中的 `Qwen2_5OmniToken2WavDiTModel.sample()` 调用
- 确保 DiT 模型的 `repeat_interleave` 操作（codec → mel 展开）不会因长度不匹配而出错

## 总结

`audio_length.py` 是一个轻量级的工具模块，解决了音频合成流程中 codec token 数量与梅尔频谱帧数之间的对齐问题。其设计关注边界条件处理（零长度、极小的 max_mel_frames 等），确保流水线的鲁棒性。
