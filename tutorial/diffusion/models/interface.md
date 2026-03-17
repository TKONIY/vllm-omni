# `interface.py` -- 模型能力接口协议定义

## 文件概述

`interface.py` 定义了一组基于 Python `Protocol` 的运行时可检查接口，用于声明扩散模型所支持的输入/输出模态（图像输入、音频输入、音频输出）。Pipeline 实现类通过继承这些协议来声明自身能力，引擎层可在运行时通过 `isinstance` 检查来做模态路由。

**文件路径**: `vllm_omni/diffusion/models/interface.py`

## 关键代码解析

### 图像输入支持协议

```python
@runtime_checkable
class SupportImageInput(Protocol):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"  # Default color format
```

声明模型支持图像作为输入（如图生图、图像编辑场景）。`color_format` 指定默认颜色格式为 RGB。

### 音频输入支持协议

```python
@runtime_checkable
class SupportAudioInput(Protocol):
    support_audio_input: ClassVar[bool] = True
```

声明模型支持音频作为输入。

### 音频输出支持协议

```python
@runtime_checkable
class SupportAudioOutput(Protocol):
    support_audio_output: ClassVar[bool] = True
```

声明模型支持生成音频输出（如 Stable Audio 文本生成音频）。

## 核心类/函数

| 类名 | 类型 | 说明 |
|------|------|------|
| `SupportImageInput` | Protocol | 图像输入协议，附带 `color_format` 属性 |
| `SupportAudioInput` | Protocol | 音频输入协议 |
| `SupportAudioOutput` | Protocol | 音频输出协议 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被实现 | `flux2/pipeline_flux2.py` | `Flux2Pipeline` 实现 `SupportImageInput`，支持图像输入 |
| 被实现 | `stable_audio/pipeline_stable_audio.py` | `StableAudioPipeline` 实现 `SupportAudioOutput` |
| 使用方 | 扩散引擎 (`DiffusionEngine`) | 通过 `isinstance(pipeline, SupportImageInput)` 判断模型能力 |

## 总结

`interface.py` 通过 Python 的 `Protocol` 机制为扩散模型定义了一套轻量级的能力声明接口。这种设计模式避免了强制继承关系，使各 Pipeline 可以灵活地声明自身支持的模态类型，便于引擎层进行运行时模态路由和前后处理选择。
