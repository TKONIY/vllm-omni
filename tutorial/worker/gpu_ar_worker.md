# `gpu_ar_worker.py` — 自回归 GPU Worker

## 文件概述

`gpu_ar_worker.py` 定义了 `GPUARWorker`，这是用于**自回归（AR）文本生成阶段**的 GPU Worker。它继承自 `OmniGPUWorkerBase`（提供进程级显存管理）并混入 `OmniWorkerMixin`（加载 Omni 插件），负责初始化 GPU 设备并创建 `GPUARModelRunner` 实例。

## 关键代码解析

### 类定义与继承

```python
class GPUARWorker(OmniWorkerMixin, OmniGPUWorkerBase):
    """GPU worker for autoregressive omni model stages."""
```

通过多重继承组合了三个功能：
- `OmniWorkerMixin`：加载 Omni 插件
- `OmniGPUWorkerBase`：进程级显存管理
- 间接继承 `GPUWorker`：上游的标准 Worker 生命周期

### init_device — 设备初始化

```python
@instrument(span_name="Init device")
def init_device(self):
```

这是 Worker 生命周期中最关键的方法，完整的初始化流程：

**1. DP (Data Parallel) 本地 Rank 调整**

```python
dp_local_rank = self.parallel_config.data_parallel_rank_local
tp_pp_world_size = (
    self.parallel_config.pipeline_parallel_size
    * self.parallel_config.tensor_parallel_size
)
self.local_rank += dp_local_rank * tp_pp_world_size
```

在数据并行场景下，调整 `local_rank` 以正确映射到物理 GPU。计算公式：`DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK`。

**2. 设备设置**

```python
self.device = torch.device(f"cuda:{self.local_rank}")
current_platform.set_device(self.device)
current_platform.check_if_supports_dtype(self.model_config.dtype)
```

**3. 分布式环境初始化**

```python
init_worker_distributed_environment(
    self.vllm_config, self.rank,
    self.distributed_init_method, self.local_rank,
    current_platform.dist_backend,
)
```

在获取显存快照之前初始化 NCCL，确保 NCCL 缓冲区的显存使用已被计入。

**4. 显存快照**

```python
gc.collect()
torch.cuda.empty_cache()
self.init_snapshot = MemorySnapshot(device=self.device)
self.requested_memory = request_memory(init_snapshot, self.cache_config)
```

**5. ModelRunner 创建**

```python
# 强制使用 v1 model runner（v2 尚未支持 Omni hooks）
if self.use_v2_model_runner:
    logger.warning("OMNI GPUARWorker forces v1 model runner for omni hooks.")
    self.use_v2_model_runner = False

self.model_runner = GPUARModelRunner(self.vllm_config, self.device)
```

注意这里强制使用 v1 model runner，因为 v2 model runner 还没有集成 Omni 的自定义 hooks。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `GPUARWorker` | 类 | 自回归阶段的 GPU Worker |
| `init_device()` | 方法 | 初始化 GPU 设备、分布式环境、ModelRunner |

## 与其他模块的关系

- **继承** `OmniGPUWorkerBase`（`base.py`）：获得进程级显存管理
- **混入** `OmniWorkerMixin`（`mixins.py`）：自动加载 Omni 插件
- **创建** `GPUARModelRunner`（`gpu_ar_model_runner.py`）：实际的模型推理执行器
- **被引擎使用**：由 vLLM-Omni 的阶段配置系统根据 `worker_cls` 字段实例化

## 总结

`GPUARWorker` 是自回归推理阶段的"外壳"，负责完成设备级别的初始化工作（GPU 选择、分布式通信、显存快照），然后将模型推理的具体工作委托给 `GPUARModelRunner`。其 `init_device()` 方法中的初始化顺序经过精心设计：先初始化 NCCL（因为它会分配显存），再拍摄显存快照，最后创建 ModelRunner。
