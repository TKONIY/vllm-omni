# `npu_model_runner.py` -- OmniNPUModelRunner 基类

## 文件概述

`npu_model_runner.py` 定义了 `OmniNPUModelRunner`，它是 NPU 平台所有 ModelRunner 的基类。该类同时继承 `OmniGPUModelRunner`（Omni 多模态逻辑）和 `NPUModelRunner`（vllm-ascend NPU 设备操作），在两者之间建立桥梁。

## 关键代码解析

### 1. 双重继承

```python
class OmniNPUModelRunner(OmniGPUModelRunner, NPUModelRunner):
```

- `OmniGPUModelRunner`：提供 Omni 多模态推理的通用逻辑（多模态输出提取、附加信息处理等）
- `NPUModelRunner`：提供 NPU 设备上的模型加载、注意力元数据构建、CUDA/ACL Graph 管理等

### 2. 模型加载与初始化

```python
def load_model(self, *args, **kwargs) -> None:
    NPUModelRunner.load_model(self, *args, **kwargs)
    enable_sp(self.vllm_config)

    talker_mtp = getattr(self.model, "talker_mtp", None)
    if talker_mtp is not None:
        self.talker_mtp = talker_mtp
        cudagraph_mode = self.compilation_config.cudagraph_mode
        has_separate_talker = getattr(self.model, "talker", None) is not None
        if cudagraph_mode.has_full_cudagraphs() and has_separate_talker:
            self.talker_mtp = ACLGraphWrapper(talker_mtp, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
```

关键步骤：
1. 调用 `NPUModelRunner.load_model` 加载模型权重
2. 初始化序列并行缓存（避免后续 `get_current_vllm_config()` 错误）
3. 如果模型包含 `talker_mtp`（Talker 的 MTP 子模块），创建对应的缓冲区并可选地用 ACL Graph 包装

### 3. Talker MTP 缓冲区

```python
hidden_size = int(
    getattr(self.model, "mtp_hidden_size", 0) or
    getattr(self.model_config.hf_text_config, "hidden_size")
)
max_batch_size = max(self.max_num_reqs, self.compilation_config.max_cudagraph_capture_size)
self.talker_mtp_input_ids = self._make_buffer(max_batch_size, dtype=torch.int32)
self.talker_mtp_inputs_embeds = self._make_buffer(max_batch_size, hidden_size, ...)
self.last_talker_hidden = self._make_buffer(max_batch_size, hidden_size, ...)
self.text_step = self._make_buffer(max_batch_size, hidden_size, ...)
```

为 Talker MTP 推理预分配 GPU 缓冲区，避免运行时反复分配内存。

### 4. _dummy_run 方法

该方法用于内存估算（profiling）和 ACL Graph 捕获，是模型初始化阶段的关键步骤。NPU 实现相比 GPU 版本有以下差异：
- 使用 `set_ascend_forward_context` 替代 CUDA 的 forward context
- 使用 `AscendAttentionState` 管理注意力状态
- 支持 Talker MTP 的 dummy 预热

### 5. _model_forward 方法

```python
def _model_forward(self, num_tokens_padded, input_ids=None, positions=None, ...):
    model_kwargs_extra = self._build_model_kwargs_extra()
    model_output = self.model(
        input_ids=input_ids, positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs, **model_kwargs_extra,
    )

    if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
        model_output = self.model.make_omni_output(model_output, **model_kwargs_extra)
    self._omni_last_model_output = model_output

    # NPU-specific: update ACL graph params
    if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
        update_full_graph_params(...)

    # NPU-specific: sequence parallelism all-gather
    if get_forward_context().flash_comm_v1_enabled:
        model_output = self._all_gather_hidden_states_and_aux(model_output)
```

该方法融合了 Omni 多模态逻辑和 NPU 设备特定操作。

### 6. _talker_mtp_forward 方法

```python
def _talker_mtp_forward(self, decode_req_ids, inputs_embeds):
    _cudagraph_mode, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(...)
    if not isinstance(self.talker_mtp, ACLGraphWrapper):
        _cudagraph_mode = CUDAGraphMode.NONE

    with set_ascend_forward_context(None, self.vllm_config, ...):
        req_embeds, code_predictor_codes = self.talker_mtp(
            req_input_ids, req_embeds, last_talker_hidden, text_step
        )
```

执行 Talker 的 MTP 子模块前向传播，生成 codec 代码预测。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OmniNPUModelRunner` | 类 | NPU ModelRunner 基类 |
| `load_model()` | 方法 | 加载模型并初始化 NPU 相关资源 |
| `_dummy_run()` | 方法 | 内存估算和图捕获 |
| `_model_forward()` | 方法 | 融合 Omni 和 NPU 的模型前向传播 |
| `_talker_mtp_forward()` | 方法 | Talker MTP 子模块前向传播 |

## 与其他模块的关系

- **继承**：`OmniGPUModelRunner`（Omni 通用逻辑）+ `NPUModelRunner`（vllm-ascend）
- **子类**：`NPUARModelRunner`、`NPUGenerationModelRunner`
- **NPU 依赖**：`vllm_ascend` 的 `set_ascend_forward_context`、`ACLGraphWrapper`、`update_cos_sin` 等
- **Omni 依赖**：`OmniOutput`、`OmniGPUModelRunner` 的多模态输出处理

## 总结

`OmniNPUModelRunner` 是一个关键的桥梁类，通过双重继承将 Omni 多模态推理逻辑与 NPU 硬件操作能力结合在一起。它处理了模型加载、Talker MTP 初始化、ACL Graph 包装、序列并行等 NPU 专有逻辑，为下游的 AR 和 Generation ModelRunner 提供了统一基础。
