# `ipc.py` — POSIX 共享内存张量传输

## 文件概述

`ipc.py` 提供了基于 POSIX 共享内存的张量传输工具，用于在 GPU Worker 和 Scheduler 之间高效传递大张量数据。当张量大小超过阈值（1MB）时，将张量数据拷贝到共享内存段中，仅通过消息队列传递轻量级的元数据句柄，避免了大数据序列化的开销。

## 关键代码解析

### 张量序列化到共享内存

```python
_SHM_TENSOR_THRESHOLD = 1_000_000  # 1 MB

def _tensor_to_shm(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu().contiguous()
    arr = tensor.numpy()
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf[:arr.nbytes])
    np.copyto(shm_arr, arr)
    handle = {
        "__tensor_shm__": True,
        "name": shm.name,
        "shape": list(tensor.shape),
        "torch_dtype": str(tensor.dtype),
        "numpy_dtype": str(arr.dtype),
        "nbytes": arr.nbytes,
    }
    shm.close()
    return handle
```

流程：张量 -> CPU contiguous -> numpy -> 拷贝到共享内存 -> 返回元数据句柄。共享内存段在发送端关闭 fd 但不 unlink，由接收端负责清理。

### 张量反序列化

```python
def _tensor_from_shm(handle: dict[str, Any]) -> torch.Tensor:
    shm = shared_memory.SharedMemory(name=handle["name"])
    try:
        arr = np.ndarray(handle["shape"], dtype=np_dtype, buffer=shm.buf[:handle["nbytes"]])
        tensor = torch.from_numpy(arr.copy())
    finally:
        shm.close()
        shm.unlink()  # 释放共享内存段
    return tensor
```

### DiffusionOutput 的打包/解包

```python
def pack_diffusion_output_shm(output: DiffusionOutput) -> DiffusionOutput:
    # 将 output.output 和 output.trajectory_latents 中的大张量替换为 SHM 句柄

def unpack_diffusion_output_shm(output: DiffusionOutput) -> DiffusionOutput:
    # 从 SHM 句柄恢复张量
```

这两个函数对 `DiffusionOutput` 中的 `output` 和 `trajectory_latents` 字段进行处理，仅当张量大小超过阈值时才使用共享内存。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_tensor_to_shm` | 函数 | 将张量拷贝到 POSIX 共享内存，返回元数据句柄 |
| `_tensor_from_shm` | 函数 | 从共享内存句柄恢复张量，并释放共享内存 |
| `pack_diffusion_output_shm` | 函数 | 将 `DiffusionOutput` 中的大张量替换为 SHM 句柄 |
| `unpack_diffusion_output_shm` | 函数 | 从 SHM 句柄恢复 `DiffusionOutput` 中的张量 |
| `_SHM_TENSOR_THRESHOLD` | 常量 | 使用共享内存的张量大小阈值，默认 1MB |

## 与其他模块的关系

- 被 `worker/diffusion_worker.py` 的 `WorkerProc.return_result` 调用 `pack_diffusion_output_shm` 进行打包
- 被 `scheduler.py` 的 `Scheduler.add_req` 调用 `unpack_diffusion_output_shm` 进行解包
- 依赖 `data.py` 中的 `DiffusionOutput` 数据结构

## 总结

`ipc.py` 通过 POSIX 共享内存实现了零拷贝式的大张量跨进程传输优化。相比通过消息队列直接序列化大张量，共享内存方案显著降低了 Worker 与 Scheduler 之间的通信开销，是多进程架构下的关键性能优化。
