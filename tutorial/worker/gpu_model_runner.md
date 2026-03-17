# `gpu_model_runner.py` — Omni 公共 ModelRunner 基类

## 文件概述

`gpu_model_runner.py` 定义了 `OmniGPUModelRunner`，它继承自上游 vLLM 的 `GPUModelRunner`，是 `GPUARModelRunner` 和 `GPUGenerationModelRunner` 的共同基类。该文件是整个 worker 模块中**最大、最核心**的文件，包含了 Omni 特有的：

- 中间缓冲区管理（`model_intermediate_buffer`）
- 多模态输出提取
- 自定义预处理/后处理流水线
- M-RoPE 位置编码修正
- Talker MTP（多 token 预测）前向传播
- FA3 元数据缓冲区修复

## 关键代码解析

### 初始化与中间缓冲区

```python
class OmniGPUModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_intermediate_buffer: dict[str, dict[str, Any]] = {}
        self._omni_num_scheduled_tokens_np: np.ndarray | None = None
        self._omni_last_model_output: object | None = None
```

`model_intermediate_buffer` 是一个核心数据结构，以请求 ID 为键，存储每个请求的中间状态数据（如 thinker 阶段的隐藏状态、生成长度等），用于跨步骤和跨阶段传递信息。

### FA3 元数据缓冲区修复

```python
def initialize_metadata_builders(self, kv_cache_config, kernel_block_sizes):
    super().initialize_metadata_builders(kv_cache_config, kernel_block_sizes)
    for kv_cache_group in self.attn_groups:
        for attn_group in kv_cache_group:
            for builder in attn_group.metadata_builders:
                sm = getattr(builder, "scheduler_metadata", None)
                max_num_splits = getattr(builder, "max_num_splits", 0)
                if sm is not None and max_num_splits > 1:
                    required = self.scheduler_config.max_num_seqs * max_num_splits + 1
                    if sm.shape[0] < required:
                        builder.scheduler_metadata = torch.zeros(
                            required, dtype=sm.dtype, device=sm.device,
                        )
```

修复了 FlashAttention3 在 CUDA Graph 捕获期间可能出现的缓冲区大小不足问题。上游分配了 `max_num_seqs + 1` 个条目，但 FA3 的 `get_scheduler_metadata()` 可能返回最多 `max_num_seqs * max_num_splits + 1` 个条目。

### 模型加载与 Talker MTP 初始化

```python
def load_model(self, *args, **kwargs) -> None:
    super().load_model(*args, **kwargs)
    talker_mtp = getattr(self.model, "talker_mtp", None)
    if talker_mtp is not None:
        self.talker_mtp = talker_mtp
        # 如果模型有独立的 talker 子模块，则用 CUDAGraphWrapper 包装
        if cudagraph_mode.has_full_cudagraphs() and has_separate_talker:
            self.talker_mtp = CUDAGraphWrapper(talker_mtp, self.vllm_config,
                                                runtime_mode=CUDAGraphMode.FULL)
        # 预分配 talker MTP 的输入缓冲区
        self.talker_mtp_input_ids = self._make_buffer(max_batch_size, dtype=torch.int32)
        self.talker_mtp_inputs_embeds = self._make_buffer(...)
        self.last_talker_hidden = self._make_buffer(...)
        self.text_step = self._make_buffer(...)
```

Talker MTP 是 Omni 模型的多 token 预测组件（如 Qwen3-Omni 的"说话者"模块），这里对其进行 CUDA Graph 包装以加速推理。

### 多模态输出提取

```python
def extract_multimodal_outputs(self, hidden_states):
    if (hasattr(self.model, "have_multimodal_outputs")
        and self.model.have_multimodal_outputs
        and isinstance(hidden_states, OmniOutput)):
        text_hidden_states = hidden_states.text_hidden_states
        multimodal_outputs = hidden_states.multimodal_outputs
    elif isinstance(hidden_states, torch.Tensor):
        text_hidden_states = hidden_states
        multimodal_outputs = {}
    ...
    return text_hidden_states, multimodal_outputs
```

从模型输出中分离文本隐藏状态和多模态输出（如音频编码、图像特征等），支持 `OmniOutput`、普通 Tensor、列表/元组等多种输出格式。

### 自定义预处理流水线 `_preprocess`

```python
def _preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
```

该方法是 Omni 推理流程的核心预处理环节，主要步骤：

1. **多模态编码器执行**：运行视觉/音频编码器，将多模态输入转换为 embedding
2. **Prompt Embeds 覆盖**：对有自定义 prompt_embeds 的请求，将其覆盖到对应位置
3. **自定义 preprocess 调用**：如果模型定义了 `has_preprocess`，逐请求调用 `model.preprocess()`
4. **Talker MTP 前向传播**：对 decode 请求运行 talker MTP 生成多 token 预测

### 状态更新 `_update_states`

```python
def _update_states(self, scheduler_output):
    # 1. 移除已完成的请求
    for req_id in scheduler_output.finished_req_ids:
        self.requests.pop(req_id, None)
        self.model_intermediate_buffer.pop(req_id, None)

    # 2. 添加新请求并解码附加信息
    for new_req_data in scheduler_output.scheduled_new_reqs:
        # 解码 prompt_embeds
        # 解码 additional_information
        # 初始化 M-RoPE 位置

    # 3. 更新运行中请求的状态
    # 4. 压缩批次并刷新元数据
```

该方法在每个推理步骤前更新持久化批次状态，Omni 扩展了以下内容：
- 清理 `model_intermediate_buffer` 中已完成请求的数据
- 解码 `additional_information` 载荷并存储到中间缓冲区
- 支持 M-RoPE 和 XD-RoPE 位置初始化

### 模型前向传播注入

```python
def _model_forward(self, input_ids=None, positions=None, ...):
    model_kwargs_extra = self._build_model_kwargs_extra()
    model_output = super()._model_forward(
        ..., **model_kwargs, **model_kwargs_extra,
    )
    if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
        model_output = self.model.make_omni_output(model_output, **model_kwargs_extra)
    self._omni_last_model_output = model_output
    return model_output
```

在调用上游的 `_model_forward` 前，注入 `model_intermediate_buffer` 和 `runtime_additional_information` 等 Omni 特有的参数。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniGPUModelRunner` | 类 | Omni 公共 ModelRunner 基类 |
| `initialize_metadata_builders()` | 方法 | 修复 FA3 scheduler_metadata 缓冲区大小 |
| `load_model()` | 方法 | 加载模型并初始化 Talker MTP |
| `_init_mrope_positions()` | 方法 | 初始化多模态 M-RoPE 位置编码 |
| `_calc_mrope_positions()` | 方法 | 计算步骤级 M-RoPE 位置（含预计算修正） |
| `_update_states()` | 方法 | 更新持久化批次状态，管理中间缓冲区 |
| `extract_multimodal_outputs()` | 方法 | 从模型输出中分离文本和多模态数据 |
| `_preprocess()` | 方法 | Omni 自定义预处理流水线 |
| `_model_forward()` | 方法 | 注入 Omni kwargs 后调用上游前向传播 |
| `_dummy_run()` | 方法 | 用于显存 profiling 和 CUDA Graph 捕获的虚拟运行 |
| `_update_intermediate_buffer()` | 方法 | 更新请求级中间缓冲区（CPU/GPU） |
| `_gather_runtime_additional_information()` | 方法 | 按批次顺序收集每请求的中间信息 |
| `_process_additional_information_updates()` | 方法 | 模型后处理，更新中间缓冲区 |
| `_talker_mtp_forward()` | 方法 | Talker MTP 多 token 预测前向传播 |

## 与其他模块的关系

- **被继承**：`GPUARModelRunner` 和 `GPUGenerationModelRunner` 均继承此类
- **上游继承**：继承 `vllm.v1.worker.gpu_model_runner.GPUModelRunner`
- **依赖** `OmniOutput`：来自 `vllm_omni.model_executor.models.output_templates`
- **依赖** `MRotaryEmbedding`：Omni 自定义的 M-RoPE 实现
- **依赖** `CUDAGraphWrapper`：用于包装 Talker MTP 以支持 CUDA Graph 加速

## 总结

`OmniGPUModelRunner` 是 worker 模块的基石，它在上游 `GPUModelRunner` 的基础上增加了：

1. **中间缓冲区系统**：实现跨步骤的请求级状态管理
2. **自定义预处理/后处理管线**：支持模型定义的 `preprocess()` / `postprocess()` 钩子
3. **多模态输出提取**：统一处理 OmniOutput、Tensor、列表等多种输出格式
4. **Talker MTP 集成**：支持多 token 预测模块的 CUDA Graph 加速
5. **M-RoPE 修正**：支持非线性解码位置编码（如图像生成的 2D 网格位置）

这些扩展使得 vLLM-Omni 能够在保留上游高性能特性的同时，支持复杂的多阶段多模态推理场景。
