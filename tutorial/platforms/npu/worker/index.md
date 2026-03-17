# npu/worker/ 子模块索引

## 模块概述

`npu/worker/` 包含 NPU 平台专用的 Worker 和 ModelRunner 实现。由于 Ascend NPU 的架构与 CUDA GPU 存在显著差异（如使用 ACL Graph 替代 CUDA Graph、序列并行实现不同、注意力机制实现不同），NPU 平台无法直接复用通用 GPU Worker，需要完整的平台专用实现。

## 类继承关系

```
vLLM NPUModelRunner (vllm_ascend)
        |
OmniGPUModelRunner (vllm_omni.worker)
        |
  OmniNPUModelRunner    <-- npu_model_runner.py (基础层)
       /        \
NPUARModelRunner  NPUGenerationModelRunner
       |                |
NPUARWorker       NPUGenerationWorker
```

## 文件列表

| 文件 | 说明 |
|------|------|
| `__init__.py` | 空初始化文件 |
| [npu_model_runner.py.md](./npu_model_runner.py.md) | OmniNPUModelRunner 基类 |
| [npu_ar_model_runner.py.md](./npu_ar_model_runner.py.md) | NPU 自回归 ModelRunner |
| [npu_ar_worker.py.md](./npu_ar_worker.py.md) | NPU 自回归 Worker |
| [npu_generation_model_runner.py.md](./npu_generation_model_runner.py.md) | NPU 生成 ModelRunner |
| [npu_generation_worker.py.md](./npu_generation_worker.py.md) | NPU 生成 Worker |

## Worker 与 ModelRunner 的关系

- **Worker**：负责设备初始化和 ModelRunner 的创建，是执行入口
- **ModelRunner**：负责模型前向传播、输入准备、输出处理等核心推理逻辑

每种 Worker 对应一种 ModelRunner：
- `NPUARWorker` -> `NPUARModelRunner`：用于 thinker/talker 阶段（自回归生成）
- `NPUGenerationWorker` -> `NPUGenerationModelRunner`：用于 code2wav 阶段（非自回归生成）

## NPU 与 GPU 实现的主要差异

1. **图模式**：NPU 使用 ACL Graph 替代 CUDA Graph
2. **注意力**：使用 Ascend 专用注意力实现（`set_ascend_forward_context`）
3. **旋转位置编码**：使用 `vllm_ascend.ops.rotary_embedding.update_cos_sin`
4. **序列并行**：通过 `enable_sp()` 启用 Ascend 序列并行
5. **同步**：使用 `torch.npu.synchronize()` 和 `torch.npu.Event()`
