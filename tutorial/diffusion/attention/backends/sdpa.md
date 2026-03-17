# `sdpa.py` — PyTorch SDPA 后端实现

## 文件概述

`sdpa.py` 实现了基于 PyTorch 原生 `scaled_dot_product_attention` (SDPA) 的注意力后端。SDPA 是最通用的后端，支持所有硬件平台（CUDA、XPU、HIP、NPU）和所有数据类型（包括 float32），通常作为 Flash Attention 不可用时的备选方案。

## 关键代码解析

### 1. 注意力掩码重塑

```python
SDPAMaskMode = Literal["broadcast_k", "full_qk"]

def _maybe_reshape_attn_mask(
    query, key, attn_mask=None, mask_mode: SDPAMaskMode = "broadcast_k",
):
    """
    将 2D 掩码 [batch_size, seq_len_k] 重塑为 SDPA 所需的格式：
    - broadcast_k: [batch_size, 1, 1, seq_len_k] —— 利用 SDPA 的广播机制
    - full_qk: [batch_size, 1, seq_len_q, seq_len_k] —— NPU 需要显式的完整掩码
    """
    if attn_mask is not None and torch.all(attn_mask != 0):
        attn_mask = None  # 全 1 掩码可优化为 None

    if attn_mask is not None and attn_mask.ndim == 2:
        attn_mask = attn_mask.to(torch.bool)
        if mask_mode == "full_qk":
            attn_mask = attn_mask.unsqueeze(1).expand(B, Sq, Skv).unsqueeze(1).contiguous()
        elif mask_mode == "broadcast_k":
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
    return attn_mask
```

关键设计：
- 全 1 掩码自动优化为 `None`（跳过掩码计算可加速）
- 不同平台使用不同的掩码模式：CUDA/XPU/HIP 使用广播模式，NPU 需要完整的 QK 掩码

### 2. 核心前向实现

```python
class SDPAImpl(AttentionImpl):
    def _forward_impl(self, query, key, value, attn_metadata=None,
                      mask_mode: SDPAMaskMode = "broadcast_k") -> torch.Tensor:
        attention_mask = None
        if attn_metadata:
            attention_mask = _maybe_reshape_attn_mask(query, key, attn_metadata.attn_mask, mask_mode=mask_mode)

        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=self.causal,
            scale=self.softmax_scale,
        )
        out = output.permute(0, 2, 1, 3)
        return out
```

实现细节：
- 输入布局转换：`(B, S, H, D)` → `(B, H, S, D)`（SDPA 要求 head-first 布局）
- 输出布局恢复：`(B, H, S, D)` → `(B, S, H, D)`
- 各平台 forward 方法通过 `mask_mode` 参数区分掩码处理方式

### 3. 多平台支持

```python
def forward_cuda(self, query, key, value, attn_metadata=None):
    return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")

def forward_npu(self, query, key, value, attn_metadata=None):
    return self._forward_impl(query, key, value, attn_metadata, mask_mode="full_qk")
```

CUDA/XPU/HIP 共享 `broadcast_k` 模式，NPU 使用 `full_qk` 模式。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `SDPABackend` | 类 | SDPA 后端工厂 |
| `SDPAImpl` | 类 | SDPA 的具体实现 |
| `SDPAImpl._forward_impl` | 方法 | 统一的前向计算实现，参数化掩码模式 |
| `SDPAImpl.forward_cuda/xpu/hip/npu` | 方法 | 各平台入口，指定掩码模式 |
| `_maybe_reshape_attn_mask` | 函数 | 掩码重塑工具，支持广播和完整两种模式 |
| `SDPAMaskMode` | 类型别名 | 掩码模式字面量类型 |

## 与其他模块的关系

- **`abstract.py`**：继承 `AttentionBackend` 和 `AttentionImpl`
- **`registry.py`**：以 `TORCH_SDPA` 名称注册在后端枚举中
- **`layer.py`**：作为 `sdpa_fallback` 在 float32 场景下被使用
- **`ring_pytorch_attn.py`**：Ring Attention 的 SDPA 路径间接使用类似逻辑

## 总结

`sdpa.py` 是最通用的注意力后端实现，依赖 PyTorch 原生 SDPA 算子。它支持全部四种硬件平台、所有数据类型、以及注意力掩码。通过 `mask_mode` 参数化设计优雅地处理了不同平台对掩码布局的差异要求。作为 Flash Attention 的备选方案，它在兼容性上是最佳选择。
