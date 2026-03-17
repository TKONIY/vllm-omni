# `utils.py` — LoRA 工具函数

## 文件概述

`utils.py` 提供了扩散模型 LoRA 系统使用的工具函数，包括目标模块匹配、packed 层模块扩展以及层替换的工厂函数。这些函数被 `DiffusionLoRAManager` 在加载和激活 LoRA 适配器时调用。

## 关键代码解析

### 1. 目标模块匹配

```python
def _match_target_modules(module_name: str, target_modules: list[str]) -> bool:
    import regex as re
    return any(
        re.match(rf".*\.{target_module}$", module_name) or target_module == module_name
        for target_module in target_modules
    )
```

该函数检查一个模块名称是否匹配 LoRA 配置中指定的 `target_modules` 列表。匹配规则是：模块名以 `.target_module` 结尾，或者完全等于 `target_module`。这与 vLLM 原有实现保持一致。

### 2. packed 层模块名扩展

```python
def _expand_expected_modules_for_packed_layers(
    supported_modules: set[str],
    packed_modules_mapping: dict[str, list[str]] | None,
) -> set[str]:
    expanded = set(supported_modules)
    for packed_name, sub_names in packed_modules_mapping.items():
        if packed_name in supported_modules:
            expanded.update(sub_names)
    return expanded
```

扩散模型中常使用 packed 投影（如 `to_qkv`），但 LoRA 权重文件通常按子投影命名（如 `to_q`/`to_k`/`to_v`）。此函数将这些子层名称加入到期望模块集合中，确保加载权重时不会被过滤掉。

### 3. 层替换工厂函数

```python
def from_layer_diffusion(layer, max_loras, lora_config, packed_modules_list, model_config=None):
    diffusion_lora_classes = [
        DiffusionMergedQKVParallelLinearWithLoRA,
        DiffusionQKVParallelLinearWithLoRA,
        DiffusionMergedColumnParallelLinearWithLoRA,
        DiffusionColumnParallelLinearWithLoRA,
        DiffusionRowParallelLinearWithLoRA,
        DiffusionReplicatedLinearWithLoRA,
    ]
    for lora_cls in diffusion_lora_classes:
        if lora_cls.can_replace_layer(source_layer=layer, ...):
            instance = lora_cls(layer)
            instance.create_lora_weights(max_loras, lora_config, model_config)
            return instance
    return layer
```

工厂函数按优先级尝试用各种 Diffusion LoRA 层类替换原始线性层。替换顺序很重要：优先匹配更具体的类型（如 MergedQKV），最后尝试通用类型（如 ReplicatedLinear）。如果没有匹配的替换类，返回原始层。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_match_target_modules` | 函数 | 检查模块名是否匹配 LoRA 的 `target_modules` 配置 |
| `_expand_expected_modules_for_packed_layers` | 函数 | 将 packed 层的子模块名称扩展到期望模块集合中 |
| `from_layer_diffusion` | 函数 | 层替换工厂函数，选择合适的 LoRA 层类替换原始线性层 |

## 与其他模块的关系

- **`manager.py`**：主要调用方，在 LoRA 适配器加载和层替换过程中使用这三个函数。
- **`layers/`**：`from_layer_diffusion` 使用 `layers/` 中定义的各种 Diffusion LoRA 层类。
- **vLLM**：`_match_target_modules` 的逻辑来源于 vLLM 的 `lora/model_manager.py`。

## 总结

`utils.py` 是 LoRA 子模块的工具层，提供模块匹配、名称扩展和层替换三个核心工具函数。其中 `from_layer_diffusion` 是连接管理器和具体 LoRA 层实现的桥梁，`_expand_expected_modules_for_packed_layers` 则解决了扩散模型 packed 投影层与 LoRA 权重命名不一致的问题。
