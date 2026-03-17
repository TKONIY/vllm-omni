# `platform.py` -- XPUOmniPlatform 实现

## 文件概述

`xpu/platform.py` 实现了 Intel XPU 平台的 `XPUOmniPlatform` 类。该类继承 `OmniPlatform` 和 vLLM 的 `XPUPlatform`，提供在 Intel GPU（如 Arc 系列）上运行 vllm-omni 的支持。

## 关键代码解析

### 1. 类定义

```python
class XPUOmniPlatform(OmniPlatform, XPUPlatform):
    _omni_enum = OmniPlatformEnum.XPU
```

### 2. 专用 Worker 类

```python
@classmethod
def get_omni_ar_worker_cls(cls) -> str:
    return "vllm_omni.platforms.xpu.worker.xpu_ar_worker.XPUARWorker"

@classmethod
def get_omni_generation_worker_cls(cls) -> str:
    return "vllm_omni.platforms.xpu.worker.xpu_generation_worker.XPUGenerationWorker"
```

XPU 使用自己的 Worker（基于 GPU Worker 的轻量级包装）。

### 3. 注意力后端

```python
@classmethod
def get_diffusion_attn_backend_cls(cls, selected_backend, head_size):
    if selected_backend is not None:
        backend = DiffusionAttentionBackendEnum[selected_backend.upper()]
        return backend.get_path()

    logger.info("Defaulting to diffusion attention backend SDPA")
    return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
```

XPU 平台默认使用 TORCH_SDPA 后端，不尝试启用 Flash Attention。

### 4. 设备操作

```python
@classmethod
def get_torch_device(cls, local_rank=None):
    if local_rank is None:
        return torch.device("xpu")
    return torch.device("xpu", local_rank)

@classmethod
def synchronize(cls) -> None:
    torch.xpu.synchronize()

@classmethod
def get_free_memory(cls, device=None):
    free, _ = torch.xpu.mem_get_info(device)
    return free
```

使用 `torch.xpu` API，设备类型为 `"xpu"`。

### 5. 功能限制

```python
@classmethod
def supports_torch_inductor(cls) -> bool:
    return False

@classmethod
def get_device_version(cls) -> str | None:
    return None
```

XPU 当前不支持 `torch.compile` inductor 后端，且没有类似 CUDA 版本号的概念。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `XPUOmniPlatform` | 类 | Intel XPU 平台实现 |
| `get_omni_ar_worker_cls()` | 方法 | 返回 XPU AR Worker |
| `get_omni_generation_worker_cls()` | 方法 | 返回 XPU Generation Worker |
| `get_diffusion_attn_backend_cls()` | 方法 | 默认使用 SDPA |
| `supports_torch_inductor()` | 方法 | 返回 False |

## 与其他模块的关系

- **继承**：`OmniPlatform` + `vllm.platforms.xpu.XPUPlatform`
- **Worker**：`xpu/worker/` 下的轻量级包装 Worker
- **阶段配置**：`xpu/stage_configs/` 目录

## 总结

`XPUOmniPlatform` 是一个相对简洁的平台实现，默认使用 SDPA 注意力后端，不支持 Flash Attention 和 torch.compile。它主要面向 Intel Arc 系列 GPU，通过 vLLM 原生 `XPUPlatform` 获取设备操作能力。
