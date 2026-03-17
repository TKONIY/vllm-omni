# `diffusion_model_runner.py` — 模型运行器

## 文件概述

`diffusion_model_runner.py` 实现了 `DiffusionModelRunner`，负责扩散模型的加载、编译、缓存配置和推理执行。它遵循 vLLM 的 AR（自回归）架构模式，将所有模型相关操作集中在 Runner 中，Worker 只负责基础设施。

## 关键代码解析

### DiffusionModelRunner 初始化

```python
class DiffusionModelRunner:
    def __init__(self, vllm_config, od_config, device):
        self.vllm_config = vllm_config
        self.od_config = od_config
        self.device = device
        self.pipeline = None
        self.cache_backend = None
        self.offload_backend = None
        self.kv_transfer_manager = OmniKVTransferManager.from_od_config(od_config)
```

Runner 管理了 pipeline（模型）、cache_backend（缓存加速）、offload_backend（CPU 卸载）和 KV Transfer Manager（跨阶段 KV 缓存传输）。

### load_model — 模型加载流程

```python
def load_model(self, memory_pool_context_fn=None, load_format=None, custom_pipeline_name=None):
    # 1. 确定加载设备（CPU 卸载时在 CPU 上加载）
    load_device = "cpu" if od_config.enable_cpu_offload else str(self.device)

    # 2. 通过 DiffusersPipelineLoader 加载模型
    with get_memory_context():
        with DeviceMemoryProfiler() as m:
            self.pipeline = model_loader.load_model(...)

    # 3. 应用 CPU 卸载
    self.offload_backend = get_offload_backend(self.od_config, device=self.device)
    if self.offload_backend:
        self.offload_backend.enable(self.pipeline)

    # 4. 应用 torch.compile（区域编译）
    if not self.od_config.enforce_eager:
        self._compile_transformer("transformer")
        self._compile_transformer("transformer_2")

    # 5. 设置缓存后端
    self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)
    if self.cache_backend:
        self.cache_backend.enable(self.pipeline)
```

加载流程包含五个阶段：模型加载 -> CPU 卸载 -> 编译 -> 缓存加速 -> 完成。

### execute_model — 推理执行

```python
def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
    # HSDP 兼容性：使用 no_grad() 而非 inference_mode()
    use_hsdp = self.od_config.parallel_config.use_hsdp
    grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()

    with grad_context:
        # 1. 接收跨阶段 KV 缓存
        self.kv_transfer_manager.receive_multi_kv_cache(req, ...)

        # 2. 设置随机数生成器
        if req.sampling_params.seed is not None:
            req.sampling_params.generator = torch.Generator(device=gen_device).manual_seed(seed)

        # 3. 刷新缓存上下文
        if self.cache_backend and self.cache_backend.is_enabled():
            self.cache_backend.refresh(self.pipeline, req.sampling_params.num_inference_steps)

        # 4. 在 forward_context 中执行 pipeline
        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            with record_function("pipeline_forward"):
                output = self.pipeline.forward(req)

        return output
```

关键设计决策：
- **HSDP 兼容**：HSDP2 的 `fully_shard` 需要 tensor version counters，因此使用 `no_grad()` 而非 `inference_mode()`
- **KV 缓存传输**：支持从 LLM 阶段接收 KV 缓存
- **缓存刷新**：每次推理前刷新 cache-dit/TeaCache 的状态

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionModelRunner` | 类 | 模型运行器，管理加载和执行 |
| `load_model` | 方法 | 完整的模型加载流程（加载/卸载/编译/缓存） |
| `execute_model` | 方法 | 执行推理前向传播 |
| `_compile_transformer` | 方法 | 对 transformer 子模块应用区域编译 |

## 与其他模块的关系

- 被 `worker/diffusion_worker.py` 的 `DiffusionWorker` 创建和调用
- 使用 `compile.py` 的 `regionally_compile` 进行编译
- 使用 `forward_context.py` 的 `set_forward_context` 设置上下文
- 使用 `model_loader/` 的 `DiffusersPipelineLoader` 加载模型
- 使用 `cache/` 和 `offloader/` 模块进行优化

## 总结

`DiffusionModelRunner` 集中管理了扩散模型的所有模型相关操作。加载流程涵盖了模型实例化、CPU 卸载、torch.compile 编译和缓存加速的完整链路。推理执行则处理了 KV 缓存传输、随机种子管理、缓存刷新和前向传播等环节。
