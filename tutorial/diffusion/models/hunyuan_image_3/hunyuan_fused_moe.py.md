# `hunyuan_fused_moe.py` — HunyuanImage3 平台自适应融合 MoE 层

## 文件概述

本文件实现了 HunyuanImage3 中使用的融合 MoE（Mixture of Experts）层 `HunyuanFusedMoE`。采用工厂模式设计，通过平台自适应机制在不同硬件平台上选择最优的 MoE 实现。

## 关键代码解析

### 1. 默认实现

```python
class HunyuanFusedMoEDefault(SharedFusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs):
        super().__init__(prefix=prefix, **kwargs)
        self._init_hook_handle = self.register_forward_pre_hook(
            self._initialize_kernel_hook, with_kwargs=True
        )

    def _initialize_kernel_hook(self, module, args, kwargs):
        if self.quant_method:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()
```

默认实现继承自 vLLM 的 `SharedFusedMoE`，通过 forward pre-hook 延迟初始化量化内核（在首次前向传播时触发一次后自动移除）。

### 2. 工厂类（平台自适应）

```python
class HunyuanFusedMoE:
    def __new__(cls, *, prefix: str = "", **kwargs):
        op_name = "hunyuan_fused_moe"
        current_omni_platform.prepare_diffusion_op_runtime(op_name)
        impl = resolve_obj_by_qualname(
            current_omni_platform.get_diffusion_model_impl_qualname(op_name),
        )
        return impl(prefix=prefix, **kwargs)
```

`HunyuanFusedMoE` 本身不是 `nn.Module`，而是通过 `__new__` 方法根据当前硬件平台动态解析并返回对应的 MoE 实现实例。

### 3. 专家参数映射

```python
@classmethod
def make_expert_params_mapping(cls, model, ckpt_gate_proj_name, ckpt_down_proj_name,
                                ckpt_up_proj_name, num_experts, num_redundant_experts=0):
    impl = resolve_obj_by_qualname(
        current_omni_platform.get_diffusion_model_impl_qualname("hunyuan_fused_moe"),
    )
    return impl.make_expert_params_mapping(...)
```

提供权重加载时的专家参数名称映射，将检查点中的 `gate_proj`/`up_proj`/`down_proj` 映射到融合 MoE 层的参数格式。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `HunyuanFusedMoEDefault` | 类 | 默认 MoE 实现（继承 SharedFusedMoE） |
| `HunyuanFusedMoE` | 工厂类 | 平台自适应 MoE 工厂 |
| `make_expert_params_mapping` | 类方法 | 专家参数权重映射 |

## 与其他模块的关系

- **`hunyuan_image_3_transformer.py`**：`HunYuanSparseMoeBlock` 中使用 `HunyuanFusedMoE` 构建专家层
- **`vllm.model_executor.layers.fused_moe.SharedFusedMoE`**：默认实现的基类
- **`vllm_omni.platforms`**：`current_omni_platform` 提供平台检测和实现解析

## 总结

`hunyuan_fused_moe.py` 通过工厂模式和平台自适应机制，使 HunyuanImage3 的 MoE 层能够在不同硬件上自动选择最优实现，同时提供延迟初始化和专家参数映射等实用功能。
