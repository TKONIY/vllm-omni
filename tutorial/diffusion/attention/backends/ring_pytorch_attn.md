# `ring_pytorch_attn.py` — Ring PyTorch Attention 实现

## 文件概述

`ring_pytorch_attn.py` 实现了基于 PyTorch 原生 SDPA 的 Ring Attention 算法。当 Flash Attention 不可用或数据类型为 float32 时，作为 Ring Flash Attention 的备选方案。算法结构与 `ring_flash_attn.py` 相同，但使用 PyTorch 的 `scaled_dot_product_attention` 作为计算内核。

## 关键代码解析

### 1. Ring Attention 前向计算

```python
class RingAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, q, k, v, sm_scale, is_causal, op_type,
                joint_tensor_key=None, joint_tensor_value=None, joint_strategy="front"):
        comm = RingComm(group)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out, lse = None, None
        next_k, next_v = None, None

        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k = comm.send_recv(k)
                next_v = comm.send_recv(v)
                comm.commit()

            if not is_causal or step <= comm.rank:
                step_k = k
                step_v = v
                if step == 0 and joint_tensor_key is not None:
                    if joint_strategy == "front":
                        step_k = torch.cat([joint_tensor_key, step_k], dim=1)
                        step_v = torch.cat([joint_tensor_value, step_v], dim=1)
                    else:
                        step_k = torch.cat([step_k, joint_tensor_key], dim=1)
                        step_v = torch.cat([step_v, joint_tensor_value], dim=1)

                block_out, block_lse = pytorch_attn_forward(
                    q, step_k, step_v,
                    softmax_scale=sm_scale,
                    causal=is_causal and step == 0,
                    op_type=op_type,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        return out.to(q.dtype)
```

与 `ring_flash_attn.py` 的核心区别：
- 使用 `pytorch_attn_forward` 而非 Flash Attention 内核
- 支持 `op_type` 参数选择底层 PyTorch 算子（`"flash"` 或 `"efficient"`）
- 支持 float32（当 op_type 为 flash 时自动降级到 efficient）

### 2. 便捷调用接口

```python
def ring_pytorch_attn_func(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
    group=None, op_type="efficient",
    joint_tensor_key=None, joint_tensor_value=None, joint_strategy="front",
):
    return RingAttentionFunc.apply(
        group, q, k, v, softmax_scale, causal, op_type,
        joint_tensor_key, joint_tensor_value, joint_strategy,
    )
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ring_pytorch_attn_func` | 函数 | Ring PyTorch Attention 的入口函数 |
| `RingAttentionFunc` | 类 | `torch.autograd.Function` 封装，实现环形注意力循环 |
| `RingAttentionFunc.forward` | 静态方法 | 核心前向计算，使用 PyTorch SDPA 作为计算内核 |

## 与其他模块的关系

- **`ring/ring_kernels.py`**：使用 `pytorch_attn_forward` 函数
- **`ring/ring_utils.py`**：使用 `update_out_and_lse` 合并分块结果
- **`distributed/comm.py`**：使用 `RingComm` 进行环形 P2P 通信
- **`parallel/ring.py`**：在 `backend_pref == "sdpa"` 时由 `RingParallelAttention` 调用

## 总结

`ring_pytorch_attn.py` 是 Ring Attention 算法的 PyTorch SDPA 实现版本。它与 `ring_flash_attn.py` 共享相同的环形通信模式和分块累积逻辑，但使用 PyTorch 原生算子，因此具有更广泛的兼容性（特别是 float32 支持），代价是性能不如 Flash Attention。
