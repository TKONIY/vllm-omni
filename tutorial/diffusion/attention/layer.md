# `layer.py` — 扩散模型注意力层核心实现

## 文件概述

`layer.py` 定义了 `Attention` 类，它是扩散模型中注意力计算的核心入口。该类作为 `nn.Module`，统一封装了注意力后端选择、并行策略管理（Ulysses / Ring）、以及前向计算流程。它采用"策略模式"将并行通信与注意力计算内核解耦，使得添加新的并行方式无需修改核心 Attention 模块。

## 关键代码解析

### 1. 构造函数：初始化后端与并行策略

```python
class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
    ):
        super().__init__()
        self.attn_backend = get_attn_backend(-1)
        self.attn_impl_cls = self.attn_backend.get_impl_cls()
        self.attention = self.attn_impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
        )
        # 实例化 SDPA 备用后端以支持 float32
        self.sdpa_fallback = SDPABackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
        )
```

关键设计要点：
- 通过 `get_attn_backend()` 获取平台最优的注意力后端（Flash Attention / SDPA / Sage 等）
- 始终初始化一个 SDPA 备用后端，用于处理 `float32` 数据类型（Flash Attention 不支持 float32）
- Ring Attention 仅在配置了 `ring_degree > 1` 且进程组可用时启用

### 2. 前向计算：三阶段流水线

```python
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata = None,
) -> torch.Tensor:
    strategy = self._get_active_parallel_strategy()

    # 1. 预处理（通信/重分片）
    query, key, value, attn_metadata, ctx = strategy.pre_attention(query, key, value, attn_metadata)

    # 2. 内核执行（计算）
    if self.use_ring and strategy is not self._no_parallel_strategy:
        out = self._run_ring_attention(query, key, value, attn_metadata)
    else:
        out = self._run_local_attention(query, key, value, attn_metadata)

    # 3. 后处理（逆向通信）
    out = strategy.post_attention(out, ctx)

    return out
```

前向过程分为三个阶段：
1. **预处理**：并行策略对 Q/K/V 进行通信或重分片（Ulysses 做 AllToAll，Ring 做拼接）
2. **计算**：根据是否启用 Ring 选择不同的注意力内核
3. **后处理**：逆向通信恢复原始分片方式

### 3. SP 活跃状态检测

```python
def _get_active_parallel_strategy(self):
    if is_forward_context_available():
        ctx = get_forward_context()
        if not ctx.sp_active:
            return self._no_parallel_strategy
    return self.parallel_strategy
```

在某些场景下（如 Z-Image 模型的 noise_refiner/context_refiner 阶段），序列并行并未激活，此时自动回退到 `NoParallelAttention`，避免不必要的通信开销。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Attention` | `nn.Module` | 扩散模型注意力层的核心模块 |
| `Attention.__init__` | 方法 | 初始化注意力后端、备用后端、Ring Attention 和并行策略 |
| `Attention.forward` | 方法 | 三阶段前向计算：预处理 → 内核执行 → 后处理 |
| `Attention._get_active_parallel_strategy` | 方法 | 根据 SP 活跃状态选择并行策略或回退到无并行 |
| `Attention._run_local_attention` | 方法 | 本地注意力计算，支持 float32 自动降级到 SDPA |
| `Attention._run_ring_attention` | 方法 | 委托给 `RingParallelAttention` 执行环形注意力 |

## 与其他模块的关系

- **`selector.py`**：调用 `get_attn_backend()` 获取注意力后端类
- **`backends/abstract.py`**：使用 `AttentionMetadata` 数据类
- **`backends/sdpa.py`**：使用 `SDPABackend` 作为 float32 的备用后端
- **`parallel/`**：使用 `build_parallel_attention_strategy()` 构建并行策略，使用 `NoParallelAttention` 和 `RingParallelAttention`
- **`distributed/parallel_state.py`**：获取序列并行进程组
- **`forward_context.py`**：获取前向上下文中的配置信息和 SP 活跃状态

## 总结

`layer.py` 是注意力模块的核心枢纽，它巧妙地将**注意力计算内核**（后端选择）与**并行通信策略**（Ulysses / Ring）解耦。通过三阶段流水线和策略模式，新增并行方式只需实现 `ParallelAttentionStrategy` 接口，无需修改核心 `Attention` 类。同时，它还具备 float32 自动降级和 SP 非活跃状态检测等鲁棒性设计。
