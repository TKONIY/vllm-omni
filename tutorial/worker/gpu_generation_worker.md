# `gpu_generation_worker.py` — 非自回归生成 GPU Worker

## 文件概述

`gpu_generation_worker.py` 定义了 `GPUGenerationWorker`，用于**非自回归生成阶段**（如 Code2Wav 音频合成、扩散模型图像生成等）的 GPU Worker。它与 `GPUARWorker` 结构几乎完全平行，唯一区别是使用 `GPUGenerationModelRunner` 代替 `GPUARModelRunner`。

## 关键代码解析

### 类定义

```python
class GPUGenerationWorker(OmniWorkerMixin, OmniGPUWorkerBase):
    """GPU Worker for Generation model (non-autoregressive waveform generation).

    Usage in stage config:
        worker_cls: "vllm_omni.worker.gpu_generation_model_runner.GPUGenerationModelRunner"
    """
```

继承结构与 `GPUARWorker` 一致：
- `OmniWorkerMixin`：加载 Omni 插件
- `OmniGPUWorkerBase`：进程级显存管理

### init_device — 设备初始化

```python
@instrument(span_name="Init device")
def init_device(self):
```

初始化流程与 `GPUARWorker.init_device()` 完全一致：

1. 清除 `NCCL_ASYNC_ERROR_HANDLING` 环境变量
2. 计算 DP 调整后的 `local_rank`
3. 设置 CUDA 设备
4. 初始化分布式环境
5. 设置随机种子
6. 获取显存快照
7. 初始化 workspace manager
8. 强制使用 v1 model runner
9. **创建 `GPUGenerationModelRunner`**（这是与 GPUARWorker 的唯一区别）

```python
# 与 GPUARWorker 的关键区别在这里
self.model_runner = GPUGenerationModelRunner(self.vllm_config, self.device)
```

### v2 Model Runner 限制

```python
if self.use_v2_model_runner:
    logger.warning("OMNI GPUGenerationWorker forces v1 model runner for omni hooks.")
    self.use_v2_model_runner = False
```

与 AR Worker 相同，强制使用 v1 model runner。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `GPUGenerationWorker` | 类 | 非自回归生成阶段的 GPU Worker |
| `init_device()` | 方法 | 初始化 GPU 设备并创建 GPUGenerationModelRunner |

## 与其他模块的关系

- **继承** `OmniGPUWorkerBase`（`base.py`）：进程级显存管理
- **混入** `OmniWorkerMixin`（`mixins.py`）：Omni 插件加载
- **创建** `GPUGenerationModelRunner`（`gpu_generation_model_runner.py`）：非自回归模型推理
- **与 `GPUARWorker` 对应**：二者是并行的 Worker 实现，分别服务于 AR 和非 AR 阶段

## 总结

`GPUGenerationWorker` 是非自回归生成阶段的 Worker 实现。它的代码与 `GPUARWorker` 高度相似（设备初始化流程完全一致），唯一的差异在于创建了 `GPUGenerationModelRunner` 而非 `GPUARModelRunner`。这种设计使得两种类型的阶段可以共享相同的设备初始化逻辑，同时使用不同的 ModelRunner 来处理各自的推理语义。
