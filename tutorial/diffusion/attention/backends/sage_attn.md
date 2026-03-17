# `sage_attn.py` — Sage Attention 后端实现

## 文件概述

`sage_attn.py` 实现了基于 SageAttention 的注意力后端。SageAttention 是一种高效的近似注意力算法，通过量化技巧（INT8 量化 QK、FP16/FP8 累积 PV）在保持精度的同时提升计算速度。

## 关键代码解析

### 1. 后端工厂

```python
class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"
```

SageAttention 支持比 Flash Attention 更多的 head size（8 种），包括 32 和 160 等非标准尺寸。

### 2. 前向计算

```python
class SageAttentionImpl(AttentionImpl):
    def forward_cuda(self, query, key, value, attn_metadata=None) -> torch.Tensor:
        output = sageattn(
            query, key, value,
            tensor_layout="NHD",
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
        )
        return output
```

实现非常简洁，直接调用 `sageattention.sageattn` 函数：
- `tensor_layout="NHD"`：使用 (batch, seq, heads, dim) 布局
- 仅支持 CUDA 平台
- 不支持注意力掩码（`supports_attention_mask` 继承默认值 `False`）

### 3. 依赖检查

```python
try:
    from sageattention import sageattn
except ImportError:
    logger.warning(
        "SageAttentionBackend is not available. You may install sage-attention"
        " by pip install git+https://github.com/thu-ml/SageAttention.git"
    )
    raise ImportError
```

在模块导入时即检查 SageAttention 库是否可用，不可用时抛出导入错误。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `SageAttentionBackend` | 类 | Sage Attention 后端工厂 |
| `SageAttentionImpl` | 类 | Sage Attention 的具体实现 |
| `SageAttentionImpl.forward_cuda` | 方法 | 调用 sageattn 执行前向计算 |

## 与其他模块的关系

- **`abstract.py`**：继承 `AttentionBackend` 和 `AttentionImpl`
- **`registry.py`**：以 `SAGE_ATTN` 名称注册在后端枚举中
- **`ring/ring_selector.py`**：Ring Attention 中也支持 Sage Attention 的变体（SAGE_AUTO/FP16/FP8 等）

## 总结

`sage_attn.py` 提供了一个简洁的 SageAttention 后端封装。相比 Flash Attention，SageAttention 通过量化技巧获得额外的性能提升，但不支持注意力掩码。它适用于对注意力掩码无要求的场景，如标准的图像生成扩散模型。
