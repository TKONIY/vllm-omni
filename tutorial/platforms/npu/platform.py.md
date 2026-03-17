# `platform.py` -- NPUOmniPlatform 实现

## 文件概述

`npu/platform.py` 实现了华为 Ascend NPU 平台的 `NPUOmniPlatform` 类。该类继承 `OmniPlatform` 和 `vllm_ascend` 的 `NPUPlatform`，是四个平台中定制化程度最高的实现，包含专用 Worker、模型算子和扩散模型运行时准备逻辑。

## 关键代码解析

### 1. 类定义与分布式后端

```python
class NPUOmniPlatform(OmniPlatform, NPUPlatform):
    _omni_enum = OmniPlatformEnum.NPU
    dist_backend: str = "hccl"
```

NPU 使用 HCCL（Huawei Collective Communication Library）作为分布式通信后端。

### 2. 专用 Worker 类

```python
@classmethod
def get_omni_ar_worker_cls(cls) -> str:
    return "vllm_omni.platforms.npu.worker.npu_ar_worker.NPUARWorker"

@classmethod
def get_omni_generation_worker_cls(cls) -> str:
    return "vllm_omni.platforms.npu.worker.npu_generation_worker.NPUGenerationWorker"
```

NPU 不复用通用 GPU Worker，而是使用完全独立的 NPU Worker 实现。

### 3. 扩散模型算子覆盖

```python
@classmethod
def get_diffusion_model_impl_qualname(cls, op_name: str) -> str:
    if op_name == "hunyuan_fused_moe":
        return "vllm_omni.platforms.npu.models.hunyuan_fused_moe.AscendHunyuanFusedMoE"
    return super().get_diffusion_model_impl_qualname(op_name)
```

NPU 平台为 `hunyuan_fused_moe` 提供了 Ascend 专用实现，替代默认的 CUDA 实现。

### 4. 运行时准备

```python
@classmethod
def prepare_diffusion_op_runtime(cls, op_name: str, **kwargs) -> None:
    if op_name != "hunyuan_fused_moe":
        return
    from vllm_omni.platforms.npu.models.hunyuan_fused_moe import (
        prepare_hunyuan_fused_moe_runtime,
    )
    prepare_hunyuan_fused_moe_runtime()
```

NPU 平台需要在运行扩散模型前初始化 MC2 通信组等运行时资源。

### 5. 注意力后端选择

```python
@classmethod
def get_diffusion_attn_backend_cls(cls, selected_backend, head_size):
    from importlib.util import find_spec

    if selected_backend is not None:
        backend = DiffusionAttentionBackendEnum[selected_backend.upper()]
        return backend.get_path()

    if find_spec("mindiesd"):
        return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()
    return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
```

NPU 的 Flash Attention 支持依赖 `mindiesd`（MindIE SDK）包的可用性。

### 6. 设备操作

```python
@classmethod
def get_torch_device(cls, local_rank=None):
    if local_rank is None:
        return torch.device("npu")
    return torch.device("npu", local_rank)

@classmethod
def get_device_total_memory(cls, device_id: int = 0) -> int:
    device_props = torch.npu.get_device_properties(device_id)
    return device_props.total_memory
```

NPU 使用 `torch.npu` API 而非 `torch.cuda`，设备类型为 `"npu"`。额外提供了 `get_device_total_memory` 方法。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `NPUOmniPlatform` | 类 | Ascend NPU 平台实现 |
| `get_omni_ar_worker_cls()` | 方法 | 返回 NPU AR Worker |
| `get_omni_generation_worker_cls()` | 方法 | 返回 NPU Generation Worker |
| `get_diffusion_model_impl_qualname()` | 方法 | 返回 NPU 专用模型算子实现 |
| `prepare_diffusion_op_runtime()` | 方法 | 初始化扩散模型运行时 |
| `get_diffusion_attn_backend_cls()` | 方法 | 基于 mindiesd 选择注意力后端 |
| `supports_torch_inductor()` | 方法 | 返回 False（NPU 不支持） |
| `get_device_total_memory()` | 方法 | 获取设备总内存 |

## 与其他模块的关系

- **继承**：`OmniPlatform` + `vllm_ascend.platform.NPUPlatform`
- **专用 Worker**：`npu/worker/` 下的完整 Worker 栈
- **专用算子**：`npu/models/hunyuan_fused_moe.py`
- **阶段配置**：`npu/stage_configs/` 目录
- **外部依赖**：`vllm_ascend`、`mindiesd`（可选）

## 总结

`NPUOmniPlatform` 是定制化程度最高的平台实现，针对 Ascend NPU 的特殊架构提供了专用的 Worker、ModelRunner 和模型算子实现。它通过覆盖扩散模型算子和运行时准备方法，使 HunyuanFusedMoE 等模型能在 NPU 上高效运行。
