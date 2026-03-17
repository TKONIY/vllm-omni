# `ring.py` — Ring 并行注意力策略

## 文件概述

`ring.py` 实现了 `RingParallelAttention` 策略类，负责 Ring Attention 的输入准备（pre_attention）、内核调度（run_attention）和输出处理（post_attention）。Ring Attention 将序列维度分片到多个设备，通过环形 P2P 通信传递 K/V 块来实现序列并行。

## 关键代码解析

### 1. 输入准备（pre_attention）

```python
def pre_attention(self, query, key, value, attn_metadata):
    joint_tensor_query = None
    joint_strategy = "front"

    if attn_metadata is not None:
        joint_tensor_query = attn_metadata.joint_query
        joint_strategy = attn_metadata.joint_strategy

    if joint_tensor_query is not None:
        if joint_strategy == "front":
            query = torch.cat([joint_tensor_query, query], dim=1)
        else:
            query = torch.cat([query, joint_tensor_query], dim=1)

        # 注意：此处不拼接 joint_key/value
        # 它们保留在 attn_metadata 中，由 Ring 内核显式处理

    ctx = _RingCtx(name=self.name)
    return query, key, value, attn_metadata, ctx
```

关键设计：
- **joint_query 拼接到 query**：文本查询与图像查询在序列维度拼接
- **joint_key/value 不拼接到 key/value**：保留在元数据中，Ring 内核会在每步的本地块（step=0）特殊处理，作为静态前缀

### 2. 内核调度（run_attention）

```python
def run_attention(self, query, key, value, attn_metadata, softmax_scale=None, causal=False):
    # 后端选择回退链：配置 → FA3 → FA2 → SDPA
    if query.dtype == torch.float32:
        backend_pref = "sdpa"
    elif not HAS_FA3 and not HAS_FLASH_ATTN:
        backend_pref = "sdpa"

    # 提取联合张量
    joint_key, joint_value = None, None
    if attn_metadata is not None:
        joint_key = attn_metadata.joint_key
        joint_value = attn_metadata.joint_value
        joint_strategy = attn_metadata.joint_strategy

    if backend_pref == "sdpa" or backend_pref == "torch":
        return ring_pytorch_attn_func(
            query, key, value,
            softmax_scale=softmax_scale,
            causal=causal,
            group=self._sp_group.ring_group,
            op_type="efficient",
            joint_tensor_key=joint_key,
            joint_tensor_value=joint_value,
            joint_strategy=joint_strategy,
        )

    attn_type = AttnType.FA3 if HAS_FA3 else AttnType.FA
    return ring_flash_attn_func(
        query, key, value,
        softmax_scale=softmax_scale,
        causal=causal,
        group=self._sp_group.ring_group,
        attn_type=attn_type,
        joint_tensor_key=joint_key,
        joint_tensor_value=joint_value,
        joint_strategy=joint_strategy,
    )
```

后端回退链：
1. **float32** → 强制使用 SDPA（Flash Attention 不支持 float32）
2. **无 FA2/FA3** → 使用 SDPA + 警告
3. **有 FA3** → 优先 FA3（性能最优）
4. **有 FA2** → 使用 FA2

### 3. 输出处理（post_attention）

```python
def post_attention(self, attn_output, ctx):
    # Ring attention output is already correctly sharded along sequence dimension.
    return attn_output
```

Ring Attention 的输出已经沿序列维度正确分片，无需额外通信。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `RingParallelAttention` | 类 | Ring 并行注意力策略 |
| `RingParallelAttention.pre_attention` | 方法 | 拼接 joint_query，保留 joint_key/value 给内核 |
| `RingParallelAttention.run_attention` | 方法 | 选择内核（FA3→FA2→SDPA）并执行 Ring Attention |
| `RingParallelAttention.post_attention` | 方法 | 直通（Ring 输出已正确分片） |
| `_RingCtx` | 数据类 | Ring 策略的上下文（当前无额外信息） |

## 与其他模块的关系

- **`base.py`**：实现 `ParallelAttentionStrategy` 接口，继承 `ParallelAttentionContext`
- **`ring_flash_attn.py`**：调用 `ring_flash_attn_func` 执行 FA2/FA3 环形注意力
- **`ring_pytorch_attn.py`**：调用 `ring_pytorch_attn_func` 执行 SDPA 环形注意力
- **`ring_globals.py`**：检查 `HAS_FA3` 和 `HAS_FLASH_ATTN` 决定内核
- **`layer.py`**：`Attention._run_ring_attention` 调用 `run_attention`

## 总结

`ring.py` 是 Ring Attention 的策略控制层。它处理联合注意力的输入准备（query 拼接，key/value 保留给内核），实现了 FA3→FA2→SDPA 的后端回退链，并利用 Ring Attention 输出自然分片的特性简化了后处理。其设计使得 Ring Attention 的通信逻辑与 `Attention` 核心类完全解耦。
