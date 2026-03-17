# `z_image.py` — Z-Image 模型 GGUF 适配器

## 文件概述

`z_image.py` 实现了 Z-Image 模型的 GGUF 权重适配器。相比 Flux2，Z-Image 的映射较为简洁，主要涉及注意力层键名和前缀移除。

## 关键代码解析

### 1. 键名映射

```python
Z_IMAGE_KEYS_RENAME_DICT = {
    "final_layer.": "all_final_layer.2-1.",
    "x_embedder.": "all_x_embedder.2-1.",
    ".attention.qkv": ".attention.to_qkv",
    ".attention.k_norm": ".attention.norm_k",
    ".attention.q_norm": ".attention.norm_q",
    ".attention.out": ".attention.to_out.0",
    "model.diffusion_model.": "",  # 移除前缀
}
```

映射规则：
- **特殊层路由**：`final_layer` 和 `x_embedder` 被重定向到带有 `2-1` 标识的路径。
- **注意力层标准化**：将 `qkv`、`q_norm`、`k_norm`、`out` 映射到 Diffusers 的 `to_qkv`、`norm_q`、`norm_k`、`to_out.0`。
- **前缀移除**：去掉 GGUF 中的 `model.diffusion_model.` 前缀。

### 2. 兼容性检查

```python
@staticmethod
def is_compatible(od_config, model, source) -> bool:
    model_class = od_config.model_class_name or ""
    if model_class.startswith("ZImage"):
        return True
    cfg = od_config.tf_model_config
    if cfg is not None:
        model_type = str(cfg.get("model_type", "")).lower()
        if model_type in {"z_image", "zimage", "z-image"}:
            return True
    return False
```

支持多种命名变体的兼容性检查。

### 3. 权重迭代

```python
gguf_to_hf_mapper = WeightsMapper(
    orig_to_new_substr=Z_IMAGE_KEYS_RENAME_DICT,
)

def weights_iterator(self):
    weights = gguf_quant_weights_iterator(self.gguf_file)
    yield from self.gguf_to_hf_mapper.apply(weights)
```

Z-Image 不需要自定义后处理（无 adaLN 交换等），直接使用 `WeightsMapper` 即可完成映射。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ZImageGGUFAdapter` | 类 | Z-Image 模型的 GGUF 适配器 |
| `is_compatible` | 静态方法 | 检查是否为 Z-Image 模型 |
| `weights_iterator` | 方法 | 产出映射后的权重 |
| `Z_IMAGE_KEYS_RENAME_DICT` | 字典 | 键名子串替换映射 |

## 与其他模块的关系

- **`base.py`**：继承 `GGUFAdapter`，使用 `gguf_quant_weights_iterator`。
- **`__init__.py`**：在适配器工厂中注册（检查顺序在 Flux2 之前）。
- **vLLM `WeightsMapper`**：用于键名映射。

## 总结

`ZImageGGUFAdapter` 是一个简洁的 GGUF 适配器实现。与 Flux2 适配器相比，Z-Image 的键名映射更直接，不需要特殊的后处理。这体现了适配器模式的价值——不同模型的映射复杂度差异很大，但可以通过统一的接口进行封装。
