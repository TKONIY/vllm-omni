# `gpu_memory_utils.py` — NVML 进程级 GPU 显存工具

## 文件概述

`gpu_memory_utils.py` 提供了基于 NVIDIA Management Library (NVML/pynvml) 的 GPU 显存查询工具函数。这些函数用于获取**当前进程**在指定 GPU 上的显存占用，是 `OmniGPUWorkerBase` 实现进程级显存管理的基础设施。

## 关键代码解析

### 检查 NVML 可用性

```python
def is_process_scoped_memory_available() -> bool:
    try:
        nvmlInit()
        nvmlShutdown()
        return True
    except Exception:
        return False
```

尝试初始化和关闭 NVML 来判断当前环境是否支持 NVML 查询。如果不可用（如容器中没有 NVML 库），Worker 会回退到基于 profiling 的显存估算。

### 解析 CUDA_VISIBLE_DEVICES

```python
def parse_cuda_visible_devices() -> list[str | int]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices:
        return []
    result: list[str | int] = []
    for item in visible_devices.split(","):
        item = item.strip()
        try:
            result.append(int(item))
        except ValueError:
            result.append(item)  # UUID (GPU-xxx) 或 MIG ID (MIG-xxx)
    return result
```

解析环境变量 `CUDA_VISIBLE_DEVICES`，支持三种格式：
- **整数索引**：如 `0,1,2`
- **GPU UUID**：如 `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- **MIG 设备 ID**：如 `MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

### 获取设备句柄

```python
def get_device_handle(device_id: str | int):
    if isinstance(device_id, int):
        return nvmlDeviceGetHandleByIndex(device_id)
    else:
        from vllm.third_party.pynvml import nvmlDeviceGetHandleByUUID
        return nvmlDeviceGetHandleByUUID(device_id)
```

根据设备标识符类型选择合适的 NVML API。

### 核心函数：获取进程级显存

```python
def get_process_gpu_memory(local_rank: int) -> int | None:
    my_pid = os.getpid()
    visible_devices = parse_cuda_visible_devices()

    try:
        nvmlInit()
    except Exception as e:
        return None  # NVML 不可用

    try:
        # 根据 CUDA_VISIBLE_DEVICES 和 local_rank 确定物理设备
        if visible_devices and local_rank < len(visible_devices):
            device_id = visible_devices[local_rank]
            handle = get_device_handle(device_id)
        else:
            handle = nvmlDeviceGetHandleByIndex(local_rank)

        # 遍历该 GPU 上运行的计算进程，找到当前 PID
        for proc in nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == my_pid:
                return proc.usedGpuMemory
        return 0  # 当前进程未使用该 GPU
    finally:
        nvmlShutdown()
```

该函数的执行流程：

1. 获取当前进程 PID
2. 解析 `CUDA_VISIBLE_DEVICES` 确定 `local_rank` 对应的物理设备
3. 通过 `nvmlDeviceGetComputeRunningProcesses` 获取该 GPU 上所有计算进程的显存使用
4. 过滤出当前 PID 的显存占用并返回

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `is_process_scoped_memory_available()` | 函数 | 检查 NVML 是否可用 |
| `parse_cuda_visible_devices()` | 函数 | 解析 CUDA_VISIBLE_DEVICES 环境变量 |
| `get_device_handle()` | 函数 | 获取 NVML 设备句柄（支持索引和 UUID） |
| `get_process_gpu_memory()` | 函数 | 获取当前进程在指定 GPU 上的显存占用（字节） |

## 与其他模块的关系

- **被调用方**：`base.py` 中的 `OmniGPUWorkerBase.determine_available_memory()` 调用这些函数
- **依赖**：`vllm.third_party.pynvml`，即 vLLM 内置的 NVIDIA Management Library Python 绑定

## 总结

`gpu_memory_utils.py` 是一个小而关键的工具模块，通过 NVML API 实现了按进程精确查询 GPU 显存的能力。它正确处理了 `CUDA_VISIBLE_DEVICES` 的各种配置格式（索引、UUID、MIG），并在 NVML 不可用时提供了优雅的降级路径（返回 `None`），由调用方决定回退策略。
