# `manager.py` — LoRA 适配器管理器

## 文件概述

`manager.py` 实现了 `DiffusionLoRAManager` 类，这是扩散模型中 LoRA 适配器的核心管理器。它复用了 vLLM 原有的 LoRA 基础设施，但针对扩散模型推理场景进行了适配，支持 LRU 缓存管理、动态层替换、packed（融合）投影处理以及适配器的激活/去激活。

## 关键代码解析

### 1. 初始化与支持模块计算

```python
class DiffusionLoRAManager:
    _VALID_MAX_RANKS: list[int] = sorted(get_args(MaxLoRARanks))

    def __init__(self, pipeline, device, dtype, max_cached_adapters=1, lora_path=None, lora_scale=1.0):
        self._supported_lora_modules = self._compute_supported_lora_modules()
        self._packed_modules_mapping = self._compute_packed_modules_mapping()
        self._expected_lora_modules = _expand_expected_modules_for_packed_layers(...)
```

初始化阶段会扫描 pipeline 中的所有线性层，提前缓存支持的 LoRA 模块后缀名。这一步必须在层替换之前进行，因为替换后原始的 `LinearBase` 会变成 `BaseLayerWithLoRA` 的子模块 `base_layer`，导致后续扫描返回错误的结果。

### 2. packed 模块映射

```python
def _compute_packed_modules_mapping(self) -> dict[str, list[str]]:
    for module in self.pipeline.modules():
        derived = _derive_from_stacked_params_mapping(
            getattr(module, "stacked_params_mapping", None)
        )
```

扩散模型中经常使用融合投影层（如 `to_qkv`、`w13`），而 LoRA 权重通常按逻辑子投影保存（如 `to_q`/`to_k`/`to_v`）。该方法从模型的 `stacked_params_mapping` 中自动推导 packed 层到子层的映射关系。

### 3. LRU 缓存管理

```python
def set_active_adapter(self, lora_request, lora_scale=1.0):
    if adapter_id not in self._registered_adapters:
        self.add_adapter(lora_request)
    else:
        self._touch_adapter_info(adapter_id)
    self._activate_adapter(adapter_id, lora_scale)

def _evict_for_new_adapter(self):
    while len(self._registered_adapters) > (self.max_cached_adapters - 1):
        evict_candidates = [aid for aid in self._adapter_access_order.keys()
                           if aid not in self._pinned_adapters]
        lru_adapter_id = evict_candidates[0]
        self.remove_adapter(lru_adapter_id)
```

管理器使用 `OrderedDict` 跟踪适配器的访问顺序，当缓存满时淘汰最近最少使用（LRU）的适配器。被 pin 的适配器不会被淘汰。

### 4. 层替换与 LoRA 激活

```python
def _replace_layers_with_lora(self, peft_helper):
    for component_name in ("transformer", "transformer_2", "dit"):
        for module_name, module in component.named_modules():
            lora_layer = from_layer_diffusion(
                layer=module, max_loras=1, lora_config=lora_config,
                packed_modules_list=packed_modules_list, model_config=None,
            )
            if lora_layer is not module and isinstance(lora_layer, BaseLayerWithLoRA):
                replace_submodule(component, module_name, lora_layer)
```

遍历 pipeline 中的 transformer 组件，将匹配的线性层替换为带有 LoRA 的封装层。替换时会检查 `target_modules` 配置，支持精确匹配和正则表达式。

### 5. 适配器权重激活

```python
def _activate_adapter(self, adapter_id, scale):
    for full_module_name, lora_layer in self._lora_modules.items():
        lora_weights = self._get_lora_weights(lora_model, full_module_name)
        # 处理 PackedLoRALayerWeights（如 QKV 融合投影）
        # 处理 fused weights（按 output_slices 切分 B 矩阵）
        # 处理普通单 slice 权重
        scaled_lora_b = lora_weights.lora_b * scale
        lora_layer.set_lora(index=0, lora_a=lora_weights.lora_a, lora_b=scaled_lora_b)
```

激活适配器时，将 LoRA 权重设置到各个替换层中。支持三种情况：packed 权重（多 slice）、fused 权重（需要切分 B 矩阵）和普通单 slice 权重。外部 scale 参数会应用到 `lora_b` 上。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionLoRAManager` | 类 | LoRA 适配器管理器，支持 LRU 缓存、动态层替换和适配器切换 |
| `__init__` | 方法 | 初始化管理器，可选在初始化时加载静态 LoRA |
| `set_active_adapter` | 方法 | 设置当前激活的 LoRA 适配器，自动处理加载和缓存 |
| `add_adapter` | 方法 | 将新适配器加入缓存（不激活） |
| `remove_adapter` | 方法 | 从缓存中移除适配器 |
| `pin_adapter` | 方法 | 将适配器标记为不可淘汰 |
| `_replace_layers_with_lora` | 方法 | 将 pipeline 中的线性层替换为 LoRA 封装层 |
| `_activate_adapter` | 方法 | 将指定适配器的权重设置到所有 LoRA 层中 |
| `_ensure_max_lora_rank` | 方法 | 确保 LoRA 缓冲区能容纳指定 rank 的适配器 |
| `_evict_for_new_adapter` | 方法 | LRU 淘汰策略实现 |

## 与其他模块的关系

- **`lora/utils.py`**：使用其中的 `_match_target_modules`、`_expand_expected_modules_for_packed_layers` 和 `from_layer_diffusion` 函数。
- **`lora/layers/`**：通过 `from_layer_diffusion` 间接使用各种 Diffusion LoRA 层类。
- **vLLM LoRA 基础设施**：复用 `LoRAModel`、`LoRARequest`、`PEFTHelper`、`BaseLayerWithLoRA` 等核心组件。
- **扩散 pipeline**：被 pipeline 调用来管理推理过程中的 LoRA 适配。

## 总结

`DiffusionLoRAManager` 是扩散模型 LoRA 系统的核心枢纽。它将 vLLM 的 LoRA 基础设施适配到扩散模型场景，处理了扩散模型特有的 packed 投影层映射、动态层替换和多种权重格式。LRU 缓存机制使得在有限显存下可以高效地切换多个 LoRA 适配器。
