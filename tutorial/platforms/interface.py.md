# `interface.py` -- OmniPlatform 抽象基类定义

## 文件概述

`interface.py` 定义了 vllm-omni 平台抽象层的核心接口。它包含 `OmniPlatformEnum` 枚举和 `OmniPlatform` 抽象基类，是所有具体平台实现（CUDA、ROCm、NPU、XPU）必须遵循的契约。该文件是整个 platforms 模块的架构基石。

## 关键代码解析

### 1. 平台枚举

```python
class OmniPlatformEnum(Enum):
    """Enum for supported Omni platforms."""
    CUDA = "cuda"
    ROCM = "rocm"
    NPU = "npu"
    XPU = "xpu"
    UNSPECIFIED = "unspecified"
```

`UNSPECIFIED` 用于未检测到任何硬件时的回退场景。

### 2. OmniPlatform 抽象基类

```python
class OmniPlatform(Platform):
    """
    Abstract base class for vllm-omni Platform.
    Inherits from vLLM's Platform and adds Omni-specific interfaces.
    """
    _omni_enum: OmniPlatformEnum
```

`OmniPlatform` 继承自 vLLM 原生的 `Platform` 类，在其基础上扩展了多模态推理相关的接口。

### 3. 平台类型判断方法

```python
def is_npu(self) -> bool:
    return self._omni_enum == OmniPlatformEnum.NPU

def is_xpu(self) -> bool:
    return self._omni_enum == OmniPlatformEnum.XPU
```

每个方法通过比较内部 `_omni_enum` 属性来判断当前平台类型，供上层代码进行平台特定分支处理。

### 4. Omni 专用抽象接口

这些方法是 OmniPlatform 相对于 vLLM Platform 新增的核心接口：

```python
@classmethod
def get_omni_ar_worker_cls(cls) -> str:
    raise NotImplementedError

@classmethod
def get_omni_generation_worker_cls(cls) -> str:
    raise NotImplementedError

@classmethod
def get_default_stage_config_path(cls) -> str:
    raise NotImplementedError
```

- `get_omni_ar_worker_cls`：返回自回归（Autoregressive）Worker 的全限定类名，用于 thinker/talker 阶段。
- `get_omni_generation_worker_cls`：返回生成 Worker 的全限定类名，用于 code2wav 等非自回归阶段。
- `get_default_stage_config_path`：返回该平台的默认阶段配置文件路径。

### 5. 扩散模型相关接口

```python
@classmethod
def get_diffusion_model_impl_qualname(cls, op_name: str) -> str:
    if op_name == "hunyuan_fused_moe":
        return "vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe.HunyuanFusedMoEDefault"
    raise NotImplementedError(f"Unsupported diffusion model op: {op_name}")

@classmethod
def get_diffusion_attn_backend_cls(cls, selected_backend: str | None, head_size: int) -> str:
    raise NotImplementedError
```

- `get_diffusion_model_impl_qualname`：根据算子名称返回平台对应的扩散模型实现类。基类提供默认实现，NPU 等平台可覆盖。
- `get_diffusion_attn_backend_cls`：选择扩散模型注意力后端（Flash Attention / SDPA 等）。

### 6. 设备操作接口

```python
@classmethod
def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
    raise NotImplementedError

@classmethod
def synchronize(cls) -> None:
    raise NotImplementedError

@classmethod
def get_free_memory(cls, device: torch.device | None = None) -> int:
    raise NotImplementedError
```

这些接口封装了不同硬件的设备操作 API（如 `torch.cuda.synchronize()` vs `torch.npu.synchronize()`）。

### 7. UnspecifiedOmniPlatform

```python
class UnspecifiedOmniPlatform(OmniPlatform):
    _omni_enum = OmniPlatformEnum.UNSPECIFIED
    device_type = ""
```

当环境中未检测到任何硬件加速器时使用的占位平台，调用其方法会触发 `NotImplementedError`。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OmniPlatformEnum` | 枚举 | 支持的平台类型枚举 |
| `OmniPlatform` | 抽象基类 | 所有平台实现的基类，定义统一接口 |
| `UnspecifiedOmniPlatform` | 类 | 无硬件时的回退占位实现 |
| `get_omni_ar_worker_cls()` | 抽象方法 | 获取自回归 Worker 类名 |
| `get_omni_generation_worker_cls()` | 抽象方法 | 获取生成 Worker 类名 |
| `get_default_stage_config_path()` | 抽象方法 | 获取阶段配置路径 |
| `get_diffusion_attn_backend_cls()` | 抽象方法 | 获取扩散注意力后端 |
| `supports_torch_inductor()` | 抽象方法 | 是否支持 torch.compile inductor 后端 |
| `get_torch_device()` | 抽象方法 | 获取 torch.device 对象 |
| `get_device_count()` | 抽象方法 | 获取可用设备数量 |
| `synchronize()` | 抽象方法 | 设备同步 |
| `get_free_memory()` | 抽象方法 | 获取设备可用内存 |

## 与其他模块的关系

- **上游依赖**：`vllm.platforms.Platform`（vLLM 原生平台基类）
- **下游实现**：`cuda/platform.py`、`rocm/platform.py`、`npu/platform.py`、`xpu/platform.py` 均继承此类
- **消费方**：Worker 创建、阶段配置加载、扩散模型后端选择等场景通过此接口获取平台特定信息

## 总结

`interface.py` 是 platforms 模块的契约定义文件。它通过 `OmniPlatform` 抽象基类规范了所有硬件平台必须实现的接口，涵盖了 Worker 类获取、设备操作、扩散模型后端选择等多个维度。该设计使得上层推理引擎代码可以完全不感知底层硬件差异，新增硬件平台支持只需实现该接口即可。
