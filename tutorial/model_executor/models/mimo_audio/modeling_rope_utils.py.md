# `modeling_rope_utils.py` — RoPE 工具函数

## 文件概述

实现多种 RoPE（旋转位置编码）计算和验证策略。从 HuggingFace Transformers 移植，为 MiMo-Audio Tokenizer 的 Transformer 层提供位置编码支持。

## 关键代码解析

### 支持的 RoPE 类型

```python
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,     # 标准 RoPE
    "linear": _compute_linear_scaling_rope_parameters,  # 线性缩放
    "dynamic": _compute_dynamic_ntk_parameters,      # 动态 NTK
    "yarn": _compute_yarn_parameters,                 # YaRN
    "longrope": _compute_longrope_parameters,         # LongRoPE
    "llama3": _compute_llama3_parameters,             # LLaMA 3 风格
}
```

### 动态更新装饰器

```python
def dynamic_rope_update(rope_forward):
    """装饰器：在 forward 前自动更新频率参数"""
    @wraps(rope_forward)
    def wrapper(self, x, position_ids):
        if "dynamic" in self.rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device)
        return rope_forward(self, x, position_ids)
```

### 旋转嵌入应用

```python
def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """应用旋转位置嵌入"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `dynamic_rope_update()` | 装饰器 | 动态 RoPE 频率更新 |
| `_compute_default_rope_parameters()` | 函数 | 标准 RoPE 频率计算 |
| `_compute_yarn_parameters()` | 函数 | YaRN RoPE 频率计算 |
| `apply_rotary_pos_emb()` | 函数 | 旋转位置嵌入应用 |
| `rotate_half()` | 函数 | 半维旋转 |

## 总结

RoPE 工具集提供了 6 种位置编码策略的统一接口，通过 `ROPE_INIT_FUNCTIONS` 字典实现策略模式。MiMo-Audio Tokenizer 默认使用 "default" RoPE。
