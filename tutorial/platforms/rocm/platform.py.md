# `platform.py` -- RocmOmniPlatform 实现

## 文件概述

`rocm/platform.py` 实现了 AMD ROCm/GPU 平台的 `RocmOmniPlatform` 类。该类继承 `OmniPlatform` 和 vLLM 的 `RocmPlatform`，提供在 AMD GPU 上运行 vllm-omni 的完整支持。

## 关键代码解析

### 1. 类定义

```python
class RocmOmniPlatform(OmniPlatform, RocmPlatform):
    _omni_enum = OmniPlatformEnum.ROCM
```

### 2. Worker 类配置

```python
@classmethod
def get_omni_ar_worker_cls(cls) -> str:
    return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

@classmethod
def get_omni_generation_worker_cls(cls) -> str:
    return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"
```

与 CUDA 平台相同，ROCm 复用通用 GPU Worker。

### 3. 扩散注意力后端

```python
@classmethod
def get_diffusion_attn_backend_cls(cls, selected_backend, head_size):
    from vllm._aiter_ops import is_aiter_found_and_supported

    compute_capability = torch.cuda.get_device_capability()
    major, minor = compute_capability
    capability = major * 10 + minor
    aiter_supported = is_aiter_found_and_supported() and 90 < capability < 100
```

ROCm 的 Flash Attention 支持与 CUDA 不同：
- 依赖 `aiter` 库而非 `flash_attn`
- 仅在 gfx942（capability 94）和 gfx950（capability 95）上支持
- 计算能力范围是 90 < capability < 100

### 4. 设备版本获取

```python
@classmethod
def get_device_version(cls) -> str | None:
    if torch.version.hip is not None:
        hip_version = torch.version.hip
        return hip_version.split("-")[0]
    return None
```

ROCm 通过 `torch.version.hip` 获取 HIP 版本号，并去除后缀信息。

### 5. 阶段配置路径

```python
@classmethod
def get_default_stage_config_path(cls) -> str:
    return "vllm_omni/platforms/rocm/stage_configs"
```

ROCm 有自己的阶段配置目录，其中包含针对 AMD GPU 优化的配置参数。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `RocmOmniPlatform` | 类 | ROCm 平台实现 |
| `get_diffusion_attn_backend_cls()` | 方法 | 基于 aiter 库可用性选择注意力后端 |
| `get_device_version()` | 方法 | 返回 HIP 版本号 |
| `supports_torch_inductor()` | 方法 | 返回 True |

## 与其他模块的关系

- **继承**：`OmniPlatform` + `vllm.platforms.rocm.RocmPlatform`
- **Worker**：复用通用 GPU Worker
- **特殊依赖**：`vllm._aiter_ops`（AMD aiter 加速库）
- **阶段配置**：`rocm/stage_configs/` 目录

## 总结

`RocmOmniPlatform` 与 CUDA 平台实现高度对称，主要差异在于 Flash Attention 的支持检测方式（使用 aiter 替代 flash_attn）和设备版本获取逻辑（HIP 版本）。ROCm 的 PyTorch 端口将 AMD GPU 映射为 `cuda` 设备类型，因此设备操作 API 仍使用 `torch.cuda.*`。
