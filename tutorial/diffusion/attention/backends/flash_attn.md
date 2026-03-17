# `flash_attn.py` — Flash Attention 后端实现

## 文件概述

`flash_attn.py` 实现了基于 Flash Attention 的注意力后端，包括 `FlashAttentionBackend`（后端工厂）和 `FlashAttentionImpl`（后端实现）。该后端是性能最优的注意力实现，支持 CUDA、XPU 和 NPU 三种硬件平台，并支持带掩码的变长序列注意力（varlen masked attention）。

## 关键代码解析

### 1. FlashAttentionBackend — 后端工厂

```python
class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"
```

该后端支持注意力掩码，接受输出缓冲区以减少内存分配，支持 5 种 head size。

### 2. CUDA 前向计算

```python
def forward_cuda(self, query, key, value, attn_metadata=None) -> torch.Tensor:
    from vllm_omni.diffusion.attention.backends.utils.fa import (
        HAS_FLASH_ATTN, flash_attn_func,
    )

    if not HAS_FLASH_ATTN:
        raise ImportError(...)

    attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None

    if attention_mask is not None and torch.any(~attention_mask):
        return self._forward_varlen_masked(query, key, value, attention_mask)

    out = flash_attn_func(query, key, value, causal=self.causal, softmax_scale=self.softmax_scale)
    return self._unwrap_flash_output(out)
```

关键逻辑：
- 检测 Flash Attention 库是否可用
- 如果存在非全 1 的注意力掩码，走变长序列路径 `_forward_varlen_masked`
- 否则直接调用 `flash_attn_func` 进行标准前向计算
- `_unwrap_flash_output` 处理 FA2 和 FA3 返回值差异（FA3 返回元组）

### 3. 变长序列掩码注意力

```python
def _forward_varlen_masked(self, query, key, value, attention_mask):
    from vllm_omni.diffusion.attention.backends.utils.fa import (
        _pad_input, _unpad_input, _upad_input, flash_attn_varlen_func,
    )

    q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
        query, key, value, attention_mask, query_length, _unpad_input
    )

    out_unpad = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k=cu_seq_lens_k,
        max_seqlen_q=max_length_q,
        max_seqlen_k=max_length_k,
        causal=self.causal,
        softmax_scale=self.softmax_scale,
    )
    out_unpad = self._unwrap_flash_output(out_unpad)
    return _pad_input(out_unpad, indices_q, query.size(0), query_length)
```

通过 unpad → varlen compute → pad 三步流程处理带掩码的序列，避免在 padding token 上浪费计算。

### 4. NPU 前向计算

```python
def forward_npu(self, query, key, value, attn_metadata=None) -> torch.Tensor:
    from mindiesd import attention_forward

    output = attention_forward(
        query, key, value,
        attn_mask=attention_mask,
        opt_mode="manual",
        op_type="fused_attn_score",
        layout="BNSD",
    )
    return output
```

NPU 平台使用华为 MindIE-SD 框架的 `attention_forward` 实现，采用 BNSD 布局。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `FlashAttentionBackend` | 类 | Flash Attention 后端工厂，声明能力和实现类 |
| `FlashAttentionImpl` | 类 | Flash Attention 的具体实现 |
| `FlashAttentionImpl.forward_cuda` | 方法 | CUDA 平台的前向计算 |
| `FlashAttentionImpl.forward_xpu` | 方法 | XPU 平台的前向计算（使用 varlen API） |
| `FlashAttentionImpl.forward_npu` | 方法 | NPU 平台的前向计算（使用 MindIE-SD） |
| `FlashAttentionImpl._forward_varlen_masked` | 方法 | 变长序列掩码注意力 |
| `FlashAttentionImpl._unwrap_flash_output` | 静态方法 | 统一 FA2/FA3 输出格式 |

## 与其他模块的关系

- **`abstract.py`**：继承 `AttentionBackend` 和 `AttentionImpl`
- **`backends/utils/fa.py`**：使用 Flash Attention 函数和 unpad/pad 工具
- **`layer.py`**：通过 `get_attn_backend()` 被 `Attention` 层实例化
- **`registry.py`**：以 `FLASH_ATTN` 名称注册在后端枚举中

## 总结

`flash_attn.py` 实现了项目中性能最优的注意力后端。它支持 CUDA/XPU/NPU 三大硬件平台，能自动处理带掩码的变长序列场景，并兼容 FA2 和 FA3 两个版本的 Flash Attention 库。通过 unpad/pad 流程避免了 padding 上的无效计算。
