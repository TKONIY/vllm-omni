# `ulysses.py` — Ulysses 并行注意力策略

## 文件概述

`ulysses.py` 实现了 `UlyssesParallelAttention` 策略类，采用 AllToAll 通信模式实现序列并行。Ulysses 策略在注意力计算前将 Q/K/V 从"序列分片"重新分片为"注意力头分片"，计算完成后逆向重新分片。它还支持联合注意力（joint attention），以及与 Ring Attention 的混合模式。

## 关键代码解析

### 1. 输入准备（pre_attention）

```python
def pre_attention(self, query, key, value, attn_metadata):
    # 1. 切分联合张量的注意力头（按 Ulysses rank）
    if joint_tensor_query is not None:
        ulysses_world_size = self._sp_group.ulysses_world_size
        ulysses_rank = self._sp_group.ulysses_rank
        attn_heads_per_ulysses_rank = joint_tensor_query.shape[-2] // ulysses_world_size

        joint_tensor_query = joint_tensor_query[
            ...,
            attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
            :,
        ]

    # 2. AllToAll：序列分片 → 头分片
    # (bs, seq_len/P, head_cnt, head_size) -> (bs, seq_len, head_cnt/P, head_size)
    query = SeqAllToAll4D.apply(self._ulysses_pg, query, self._scatter_idx, self._gather_idx, self._use_sync)
    key = SeqAllToAll4D.apply(self._ulysses_pg, key, self._scatter_idx, self._gather_idx, self._use_sync)
    value = SeqAllToAll4D.apply(self._ulysses_pg, value, self._scatter_idx, self._gather_idx, self._use_sync)

    # 3. AllToAll 后拼接联合 query
    if is_joint:
        if joint_strategy == "front":
            query = torch.cat([joint_tensor_query, query], dim=1)
        else:
            query = torch.cat([query, joint_tensor_query], dim=1)

    # 4. 如果不是混合 Ring 模式，拼接联合 key/value
    use_ring = self._sp_group.ring_world_size > 1
    if is_joint and not use_ring:
        if joint_strategy == "front":
            key = torch.cat([joint_tensor_key, key], dim=1)
            value = torch.cat([joint_tensor_value, value], dim=1)
```

Ulysses 的核心变换：
1. **联合张量头切分**：联合 Q/K/V（如文本条件）在各 rank 间复制，每个 rank 只取自己负责的头
2. **AllToAll 通信**：scatter `dim=2`（头维度），gather `dim=1`（序列维度）
3. **联合拼接**：在 AllToAll 之后拼接联合张量，保证维度一致
4. **混合模式处理**：如果同时启用 Ring，则不在此处拼接 joint_key/value（交给 Ring 内核处理）

### 2. 掩码处理

```python
if attn_metadata is not None and is_joint:
    if attn_metadata.joint_attn_mask is None and attn_metadata.attn_mask is None:
        attn_metadata.attn_mask = None
    else:
        # 补全缺失的掩码为全 1
        if attn_metadata.attn_mask is None:
            attn_metadata.attn_mask = torch.ones(...)
        elif attn_metadata.joint_attn_mask is None:
            attn_metadata.joint_attn_mask = torch.ones(...)
        # 拼接掩码
        attn_metadata.attn_mask = (
            torch.cat([attn_metadata.joint_attn_mask, attn_metadata.attn_mask], dim=1)
            if joint_strategy == "front"
            else torch.cat([attn_metadata.attn_mask, attn_metadata.joint_attn_mask], dim=1)
        )
```

联合注意力的掩码需要与查询拼接对齐：缺失的掩码自动补全为全 1。

### 3. 输出后处理（post_attention）

```python
def post_attention(self, attn_output, ctx):
    if ctx.joint_len > 0:
        # 分离联合输出和图像输出
        if ctx.joint_strategy == "front":
            output_joint = attn_output[:, :joint_len]
            output_img = attn_output[:, joint_len:]
        else:
            output_img = attn_output[:, :-joint_len]
            output_joint = attn_output[:, -joint_len:]

        # 图像部分：逆 AllToAll（头分片 → 序列分片）
        output_img = SeqAllToAll4D.apply(ctx.ulysses_pg, output_img, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync)

        # 联合部分：AllGather 恢复全部头
        gathered_joint = [torch.zeros_like(output_joint) for _ in range(dist.get_world_size(ctx.ulysses_pg))]
        dist.all_gather(gathered_joint, output_joint, group=ctx.ulysses_pg)
        output_joint = torch.cat(gathered_joint, dim=2)

        # 重组
        return torch.cat([output_joint, output_img], dim=1) if joint_strategy == "front" else ...

    # 无联合张量：标准逆 AllToAll
    return SeqAllToAll4D.apply(ctx.ulysses_pg, attn_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync)
```

后处理的两种路径：
1. **有联合张量**：图像部分做逆 AllToAll，联合部分做 AllGather 恢复所有头，然后重新拼接
2. **无联合张量**：简单的逆 AllToAll

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `UlyssesParallelAttention` | 类 | Ulysses 并行注意力策略 |
| `UlyssesParallelAttention.pre_attention` | 方法 | AllToAll 重分片 + 联合张量处理 + 掩码拼接 |
| `UlyssesParallelAttention.post_attention` | 方法 | 逆 AllToAll + AllGather 联合部分 |
| `_UlyssesCtx` | 数据类 | Ulysses 上下文，存储进程组、切分索引、联合长度等 |

## 与其他模块的关系

- **`base.py`**：继承 `ParallelAttentionContext`，实现 `ParallelAttentionStrategy` 接口
- **`distributed/comm.py`**：使用 `SeqAllToAll4D` 进行 4D 张量的 AllToAll 通信
- **`distributed/group_coordinator.py`**：使用 `SequenceParallelGroupCoordinator` 获取进程组信息
- **`factory.py`**：在 `ulysses_degree > 1` 时被创建
- **`layer.py`**：被 `Attention.forward` 使用

## 总结

`ulysses.py` 实现了基于 AllToAll 通信的 Ulysses 序列并行策略。其核心是将"序列分片、全头"转换为"全序列、头分片"进行注意力计算，然后逆向恢复。对联合注意力的支持尤为复杂：联合张量在 AllToAll 前切分头维度，在 AllToAll 后拼接，后处理时通过 AllGather 恢复全部头。它还支持与 Ring Attention 的混合模式。
