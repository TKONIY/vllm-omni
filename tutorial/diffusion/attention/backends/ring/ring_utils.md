# `ring_utils.py` — Ring Attention 分块结果合并工具

## 文件概述

`ring_utils.py` 提供了 Ring Attention 最核心的数学工具 —— 分块注意力结果的正确合并。在 Ring Attention 中，每个设备每步只计算查询与一个 K/V 块的注意力，需要通过 log-sum-exp (LSE) 技巧将多步结果正确合并为全局注意力输出。

## 关键代码解析

### 1. 核心合并公式

```python
def _update_out_and_lse(out, lse, block_out, block_lse):
    block_out = block_out.to(torch.float32)

    # 数值稳定的在线 softmax 合并
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse
```

这是在线 softmax 算法的核心公式。对于两个分块的注意力结果 `(out1, lse1)` 和 `(out2, lse2)`，合并规则为：

- `new_out = out1 - sigmoid(lse2 - lse1) * (out1 - out2)`
- `new_lse = lse1 - logsigmoid(lse1 - lse2)`

其中 `sigmoid` 和 `logsigmoid` 保证了数值稳定性，等价于经典的 max-减法技巧。

### 2. LSE 形状自适应

```python
def _update_out_and_lse(out, lse, block_out, block_lse):
    B, S, H, D = out.shape

    # Case 1: block_lse 是 3D (B, H, S) — 标准 SDPA/FA 输出
    if block_lse.dim() == 3:
        if block_lse.shape[1] == H and block_lse.shape[2] == S:
            block_lse = block_lse.transpose(1, 2).unsqueeze(-1)  # -> (B, S, H, 1)

        # Case 2: (B, S, H) — 已经是目标布局
        elif block_lse.shape[1] == S and block_lse.shape[2] == H:
            block_lse = block_lse.unsqueeze(-1)

        # Case 3: (B, H, S_pad) — 带 padding 的情况
        elif block_lse.shape[1] == H and block_lse.shape[2] >= S:
            block_lse = block_lse[:, :, :S].transpose(1, 2).unsqueeze(-1)
```

由于不同注意力内核返回的 LSE 形状各异（FA 返回 `(B, H, S)`，SDPA 返回 `(B, H, S)` 或 `(B, H, S_padded)` 等），这段代码通过穷举和推断自动将 LSE 归一化为 `(B, S, H, 1)` 形状，以匹配输出张量 `out` 的 `(B, S, H, D)` 形状。

### 3. 首次调用的初始化逻辑

```python
def update_out_and_lse(out, lse, block_out, block_lse, slice_=None):
    if out is None:
        out = block_out.to(torch.float32)
        # 根据 block_lse 的形状推断正确的 LSE 初始化
        if block_lse.dim() == 3:
            if block_lse.shape[1] == H_guess and block_lse.shape[2] == S_guess:
                lse = block_lse.transpose(1, 2).unsqueeze(-1)
            # ... 多种情况处理
    elif slice_ is not None:
        # 切片更新模式
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse
```

`update_out_and_lse` 是对外接口，处理三种情况：
1. **首次调用**（`out is None`）：直接用第一个块初始化，并处理 LSE 形状
2. **切片更新**（`slice_` 不为 None）：仅更新指定切片
3. **标准更新**：调用 `_update_out_and_lse` 合并

### 4. 变长序列 LSE 工具

```python
def flatten_varlen_lse(lse, cu_seqlens):
    """将批次化的 LSE 展平为紧凑格式。"""
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)

def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen):
    """将紧凑格式的 LSE 恢复为批次化的带 padding 格式。"""
    new_lse = torch.empty((num_seq, max_seqlen, num_head, 1), ...)
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()
```

用于变长序列（varlen）场景下 LSE 的格式转换。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `update_out_and_lse` | 函数 | 对外接口：合并分块注意力结果（含首次初始化） |
| `_update_out_and_lse` | 函数 | 核心合并逻辑：使用 sigmoid/logsigmoid 公式 |
| `flatten_varlen_lse` | 函数 | 将批次 LSE 展平为紧凑格式 |
| `unflatten_varlen_lse` | 函数 | 将紧凑 LSE 恢复为批次格式 |

## 与其他模块的关系

- **`ring_flash_attn.py`**：在环形循环中调用 `update_out_and_lse` 累积结果
- **`ring_pytorch_attn.py`**：同样在环形循环中使用
- **`ring_kernels.py`**：各内核返回的 `(out, lse)` 格式是本模块的输入

## 总结

`ring_utils.py` 实现了 Ring Attention 算法中最关键的数学操作 —— 在线 softmax 合并。通过 `sigmoid` 和 `logsigmoid` 函数保证了数值稳定性，通过大量的 LSE 形状自适应逻辑兼容了不同注意力内核的输出差异。这是 Ring Attention 能够正确工作的数学基础。
