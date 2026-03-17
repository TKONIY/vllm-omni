# `platform.py` -- CudaOmniPlatform 实现

## 文件概述

`cuda/platform.py` 实现了 NVIDIA CUDA/GPU 平台的 `CudaOmniPlatform` 类。该类同时继承 `OmniPlatform` 和 vLLM 的 `CudaPlatformBase`，获得了完整的 CUDA 设备操作能力和 Omni 多模态推理接口。作为默认平台，它是最完整的参考实现。

## 关键代码解析

### 1. 类定义与继承

```python
class CudaOmniPlatform(OmniPlatform, CudaPlatformBase):
    _omni_enum = OmniPlatformEnum.CUDA
```

双重继承：
- `OmniPlatform`：提供 Omni 多模态接口
- `CudaPlatformBase`：提供 vLLM 原生 CUDA 功能（量化支持、编译配置等）

### 2. Worker 类注册

```python
@classmethod
def get_omni_ar_worker_cls(cls) -> str:
    return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

@classmethod
def get_omni_generation_worker_cls(cls) -> str:
    return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"
```

CUDA 平台直接使用通用 GPU Worker，无需平台专用实现。

### 3. 扩散注意力后端选择

```python
@classmethod
def get_diffusion_attn_backend_cls(cls, selected_backend, head_size):
    compute_capability = cls.get_device_capability()
    compute_supported = False
    if compute_capability is not None:
        major, minor = compute_capability
        capability = major * 10 + minor
        compute_supported = 80 <= capability < 100

    packages_info = PACKAGES_CHECKER.get_packages_info()
    packages_available = packages_info.get("has_flash_attn", False)
    flash_attn_supported = compute_supported and packages_available

    if selected_backend is not None:
        backend_upper = selected_backend.upper()
        if backend_upper == "FLASH_ATTN" and not flash_attn_supported:
            return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
        return DiffusionAttentionBackendEnum[backend_upper].get_path()

    if flash_attn_supported:
        return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()
    return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
```

选择逻辑：
1. Flash Attention 需要同时满足：计算能力 8.0-9.x 且 `flash_attn` 包已安装
2. 用户指定后端时优先尊重用户选择，不满足条件时自动回退到 SDPA
3. 未指定时默认优先使用 Flash Attention

### 4. 设备操作

```python
@classmethod
def get_torch_device(cls, local_rank=None):
    if local_rank is None:
        return torch.device("cuda")
    return torch.device("cuda", local_rank)

@classmethod
def get_free_memory(cls, device=None):
    free, _ = torch.cuda.mem_get_info(device)
    return free
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `CudaOmniPlatform` | 类 | CUDA 平台实现，继承 OmniPlatform + CudaPlatformBase |
| `get_omni_ar_worker_cls()` | 方法 | 返回 GPU AR Worker 类名 |
| `get_omni_generation_worker_cls()` | 方法 | 返回 GPU Generation Worker 类名 |
| `get_diffusion_attn_backend_cls()` | 方法 | 基于计算能力选择扩散注意力后端 |
| `supports_torch_inductor()` | 方法 | 返回 True，CUDA 支持 torch.compile |
| `get_device_capability()` | 方法 | 获取 GPU 计算能力（major, minor） |
| `get_device_name()` | 方法 | 获取 GPU 设备名称 |

## 与其他模块的关系

- **继承关系**：`OmniPlatform`（Omni 接口）+ `CudaPlatformBase`（vLLM CUDA 功能）
- **Worker 引用**：`vllm_omni.worker.gpu_ar_worker` 和 `vllm_omni.worker.gpu_generation_worker`
- **扩散后端**：`vllm_omni.diffusion.attention.backends.registry`
- **阶段配置**：使用默认路径 `vllm_omni/model_executor/stage_configs`

## 总结

`CudaOmniPlatform` 是最简洁的平台实现，充分利用了 vLLM 原有的 CUDA 基础设施。由于 CUDA 是主要开发平台，它直接复用通用 GPU Worker 而不需要平台特有的 Worker/ModelRunner 实现。其扩散注意力后端选择逻辑是最完整的，考虑了计算能力和软件包可用性的双重检查。
