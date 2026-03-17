# `base.py` — GPU Worker 基类（进程级显存管理）

## 文件概述

`base.py` 定义了 `OmniGPUWorkerBase`，它是所有 vLLM-Omni GPU Worker 的基类。该类继承自上游 vLLM 的 `GPUWorker`，核心贡献是重写了 `determine_available_memory()` 方法，引入**进程级 GPU 显存追踪**，使多个推理阶段可以安全地在同一张 GPU 上并发初始化。

## 关键代码解析

### 类定义与继承

```python
class OmniGPUWorkerBase(GPUWorker):
    """Base GPU worker for vLLM-Omni with process-scoped memory accounting."""
```

继承自 `vllm.v1.worker.gpu_worker.Worker`（即上游的 `GPUWorker`），保留了标准的设备初始化、模型加载等流程。

### 显存计算核心逻辑

```python
@torch.inference_mode()
def determine_available_memory(self) -> int:
```

该方法决定了当前 Worker 可用于 KV Cache 的显存大小。算法分为两条路径：

**路径一：NVML 可用（推荐）**

```python
process_memory = (
    get_process_gpu_memory(self.local_rank)
    if is_process_scoped_memory_available() and detect_pid_host()
    else None
)

if process_memory is not None:
    self.available_kv_cache_memory_bytes = max(0, self.requested_memory - process_memory)
```

通过 NVML 获取**当前进程**占用的 GPU 显存（而非整张卡的显存），计算公式为：

```
可用KV缓存 = 请求的总显存 - 本进程已用显存
```

这种方式允许多个阶段在同一 GPU 上并行初始化，因为每个进程只看到自己的显存使用。

**路径二：回退方案（NVML 不可用时）**

```python
profiled_usage = (
    int(self.model_runner.model_memory_usage)
    + profile_result.torch_peak_increase
    + profile_result.non_torch_increase
)
self.available_kv_cache_memory_bytes = max(0, self.requested_memory - profiled_usage)
```

使用 profile 数据估算，公式为：

```
可用KV缓存 = 请求的总显存 - (模型权重 + 峰值激活 + 非Torch开销)
```

### 显式 KV Cache 大小的快速路径

```python
if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
    self.model_runner.profile_run()
    return kv_cache_memory_bytes
```

如果用户在配置中显式指定了 KV Cache 大小，则直接跳过显存计算。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniGPUWorkerBase` | 类 | 所有 Omni GPU Worker 的基类 |
| `determine_available_memory()` | 方法 | 计算可用于 KV Cache 的 GPU 显存 |

## 与其他模块的关系

- **被继承**：`GPUARWorker` 和 `GPUGenerationWorker` 均继承自 `OmniGPUWorkerBase`
- **依赖** `gpu_memory_utils.py`：调用 `get_process_gpu_memory()` 和 `is_process_scoped_memory_available()` 进行 NVML 查询
- **依赖** `vllm_omni.entrypoints.utils.detect_pid_host`：判断当前运行环境是否支持 PID 级别的显存追踪
- **上游依赖**：继承 `vllm.v1.worker.gpu_worker.Worker`，使用 `memory_profiling` 上下文管理器

## 总结

`OmniGPUWorkerBase` 解决了 vLLM-Omni 多阶段架构下同 GPU 并发初始化的显存冲突问题。通过 NVML 的进程级显存查询，每个阶段的 Worker 可以准确地知道自己用了多少显存，从而正确计算 KV Cache 容量，而不会被其他阶段的显存占用干扰。
