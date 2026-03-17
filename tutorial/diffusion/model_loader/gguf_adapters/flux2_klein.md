# `flux2_klein.py` — Flux2-Klein 模型 GGUF 适配器

## 文件概述

`flux2_klein.py` 实现了 Flux2-Klein 模型的 GGUF 权重适配器。Flux2 系列模型的 GGUF 权重使用自己的命名约定，需要映射到 Diffusers/HuggingFace 格式的参数名。该适配器还处理了 adaLN（自适应层归一化）参数的 shift/scale 交换问题。

## 关键代码解析

### 1. 键名映射字典

```python
FLUX2_TRANSFORMER_KEYS_RENAME_DICT = {
    "single_blocks.": "single_transformer_blocks.",
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    ...
}

FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP = {
    "double_blocks.": "transformer_blocks.",
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.proj": "attn.to_out.0",
    "img_attn.qkv": "attn.to_qkv",
    "txt_attn.qkv": "attn.add_kv_proj",
    ...
}

FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    ...
}
```

映射涵盖四个层面：
- **全局重命名**：输入嵌入、时间步嵌入、引导嵌入等。
- **adaLN 重命名**：`final_layer.adaLN_modulation.1` -> `norm_out.linear`。
- **双流 block**：图像/文本注意力和 MLP 的映射。
- **单流 block**：融合注意力和 MLP 的映射。

### 2. 兼容性检查

```python
@staticmethod
def is_compatible(od_config, model, source) -> bool:
    model_class = od_config.model_class_name or ""
    if model_class.startswith("Flux2"):
        return True
    cfg = od_config.tf_model_config
    if cfg is not None:
        model_type = str(cfg.get("model_type", "")).lower()
        if model_type.startswith("flux"):
            return True
    return False
```

通过 `model_class_name` 或 `tf_model_config.model_type` 判断是否为 Flux2 模型。

### 3. WeightsMapper 与自定义权重处理

```python
gguf_to_hf_mapper = WeightsMapper(
    orig_to_new_prefix=FLUX2_TRANSFORMER_KEYS_RENAME_DICT | FLUX2_TRANSFORMER_ADA_LAYER_NORM_KEY_MAP,
    orig_to_new_substr=FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP | FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP,
)

def weights_iterator(self):
    def custom_weights_adapter(weights):
        for name, weight in weights:
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight")
            if name == "norm_out.linear.weight":
                # adaLN 参数的 shift/scale 交换
                shift, scale = weight.chunk(2, dim=0)
                weight = torch.cat([scale, shift], dim=0)
            yield name, weight

    weights = gguf_quant_weights_iterator(self.gguf_file)
    weights = self.gguf_to_hf_mapper.apply(weights)
    yield from custom_weights_adapter(weights)
```

处理流程：
1. 使用 `gguf_quant_weights_iterator` 读取 GGUF 文件。
2. 使用 vLLM 的 `WeightsMapper` 进行批量键名映射。
3. 自定义后处理：将 `.scale` 后缀改为 `.weight`，并对 `norm_out.linear.weight` 执行 shift/scale 交换。

adaLN 参数交换的原因是 Flux2 的 GGUF 检查点中 `[shift, scale]` 顺序与 Diffusers 实现中 `[scale, shift]` 的预期相反。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Flux2KleinGGUFAdapter` | 类 | Flux2-Klein 模型的 GGUF 适配器 |
| `is_compatible` | 静态方法 | 检查是否为 Flux2 系列模型 |
| `weights_iterator` | 方法 | 产出映射后的权重，包含 adaLN 交换 |
| `gguf_to_hf_mapper` | 类变量 | vLLM WeightsMapper 实例，执行批量键名映射 |
| `FLUX2_TRANSFORMER_KEYS_RENAME_DICT` | 字典 | 全局键名重命名映射 |
| `FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP` | 字典 | 双流 block 键名映射 |
| `FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP` | 字典 | 单流 block 键名映射 |

## 与其他模块的关系

- **`base.py`**：继承 `GGUFAdapter`，使用 `gguf_quant_weights_iterator`。
- **`__init__.py`**：在适配器工厂中注册。
- **vLLM `WeightsMapper`**：用于高效的批量键名映射。
- **Flux2 模型实现**：映射字典与模型的参数命名一一对应。

## 总结

`Flux2KleinGGUFAdapter` 处理了 Flux2 模型 GGUF 权重到 Diffusers 格式的完整映射。主要复杂性来自：(1) Flux2 架构的双流/单流 block 有大量需要映射的键名；(2) adaLN 参数需要 shift/scale 交换。通过 `WeightsMapper` + 自定义后处理的两层架构，保持了映射逻辑的清晰和可维护性。
