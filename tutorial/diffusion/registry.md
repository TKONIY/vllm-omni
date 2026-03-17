# `registry.py` — 模型注册与初始化

## 文件概述

`registry.py` 是扩散模型的注册中心，负责管理所有支持的扩散模型类、其前后处理函数，以及模型初始化流程（包括 VAE 配置和序列并行应用）。它使用 vLLM 的 `_ModelRegistry` 实现懒加载式模型注册。

## 关键代码解析

### 模型注册表

```python
_DIFFUSION_MODELS = {
    "QwenImagePipeline": ("qwen_image", "pipeline_qwen_image", "QwenImagePipeline"),
    "WanPipeline": ("wan2_2", "pipeline_wan2_2", "Wan22Pipeline"),
    "FluxPipeline": ("flux", "pipeline_flux", "FluxPipeline"),
    # ... 20+ 模型
}

DiffusionModelRegistry = _ModelRegistry({
    model_arch: _LazyRegisteredModel(
        module_name=f"vllm_omni.diffusion.models.{mod_folder}.{mod_relname}",
        class_name=cls_name,
    )
    for model_arch, (mod_folder, mod_relname, cls_name) in _DIFFUSION_MODELS.items()
})
```

每个模型通过 `(文件夹, 模块名, 类名)` 三元组注册，支持懒加载（仅在使用时才 import）。

### initialize_model — 模型初始化

```python
def initialize_model(od_config: OmniDiffusionConfig) -> nn.Module:
    model_class = DiffusionModelRegistry._try_load_model_cls(od_config.model_class_name)
    model = model_class(od_config=od_config)
    # 配置 VAE（slicing/tiling/分布式）
    # 应用序列并行
    _apply_sequence_parallel_if_enabled(model, od_config)
    return model
```

### 前后处理函数注册

```python
_DIFFUSION_POST_PROCESS_FUNCS = {
    "QwenImagePipeline": "get_qwen_image_post_process_func",
    "WanPipeline": "get_wan22_post_process_func",
    # ...
}

_DIFFUSION_PRE_PROCESS_FUNCS = {
    "GlmImagePipeline": "get_glm_image_pre_process_func",
    "WanPipeline": "get_wan22_pre_process_func",
    # ...
}
```

前后处理函数通过函数名注册，在运行时通过 `importlib` 动态加载对应模块中的函数。

### 序列并行应用

```python
def _apply_sequence_parallel_if_enabled(model, od_config):
    sp_size = od_config.parallel_config.sequence_parallel_size
    if sp_size <= 1:
        return
    # 遍历 transformer/transformer_2/dit/unet 属性
    # 从模型获取 _sp_plan
    # 应用 SequenceParallelConfig + hooks
    apply_sequence_parallel(transformer, sp_config, plan)
```

此函数在模型加载时自动检测并应用序列并行，支持 Ulysses、Ring 及混合模式。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_DIFFUSION_MODELS` | dict | 所有注册的扩散模型映射表 |
| `DiffusionModelRegistry` | `_ModelRegistry` | 懒加载模型注册器 |
| `initialize_model` | 函数 | 初始化扩散模型（加载类、配置 VAE、应用 SP） |
| `get_diffusion_post_process_func` | 函数 | 获取模型特定的后处理函数 |
| `get_diffusion_pre_process_func` | 函数 | 获取模型特定的前处理函数 |
| `_apply_sequence_parallel_if_enabled` | 函数 | 条件性地应用序列并行 hooks |
| `_NO_CACHE_ACCELERATION` | set | 不支持缓存加速的模型集合 |

## 与其他模块的关系

- 被 `diffusion_engine.py` 调用以获取前后处理函数
- 被 `model_loader/` 调用 `initialize_model` 来实例化模型
- 引用 `hooks/sequence_parallel.py` 的 `apply_sequence_parallel` 函数
- 引用 `distributed/sp_plan.py` 获取模型的序列并行计划
- 引用 `forward_context.py` 更新 `sp_plan_hooks_applied` 状态

## 总结

`registry.py` 是模型管理的核心枢纽，将模型类、前后处理函数通过注册表统一管理，并在模型初始化时自动完成 VAE 配置和序列并行应用。新增模型只需在 `_DIFFUSION_MODELS` 和对应的前后处理字典中添加条目即可。
