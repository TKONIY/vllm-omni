# `factory.py` — 并行注意力策略工厂

## 文件概述

`factory.py` 提供了工厂函数 `build_parallel_attention_strategy`，根据当前的扩散模型配置和分布式环境自动选择最合适的并行注意力策略。

## 关键代码解析

### 策略选择逻辑

```python
def build_parallel_attention_strategy(
    *, scatter_idx: int, gather_idx: int, use_sync: bool,
) -> ParallelAttentionStrategy:
    try:
        cfg = get_forward_context().omni_diffusion_config
        p = cfg.parallel_config
    except Exception as e:
        logger.debug(f"No forward context available: {e}")
        return NoParallelAttention()

    ulysses_degree = getattr(p, "ulysses_degree", 1)
    ring_degree = getattr(p, "ring_degree", 1)

    try:
        sp_group = get_sp_group()
        if get_sequence_parallel_world_size() <= 1:
            return NoParallelAttention()
    except Exception as e:
        if ulysses_degree > 1 or ring_degree > 1:
            logger.warning(f"SP configured but group not available: {e}.")
        return NoParallelAttention()

    # 优先 Ulysses（或混合 Ulysses+Ring）
    if ulysses_degree > 1:
        return UlyssesParallelAttention(
            sp_group=sp_group,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
        )

    # 纯 Ring Attention
    if ring_degree > 1:
        return RingParallelAttention(sp_group=sp_group)

    return NoParallelAttention()
```

选择优先级：
1. **无上下文** → `NoParallelAttention`
2. **SP world_size <= 1** → `NoParallelAttention`
3. **ulysses_degree > 1** → `UlyssesParallelAttention`（也支持混合 Ulysses+Ring）
4. **ring_degree > 1** → `RingParallelAttention`
5. **其他** → `NoParallelAttention`

关键设计原则：
- 注意力内核后端选择保持在 `selector.py`，并行策略选择在此处
- 当配置了 SP 但进程组不可用时，发出警告并安全降级

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `build_parallel_attention_strategy` | 函数 | 根据配置和分布式环境构建并行策略 |

## 与其他模块的关系

- **`base.py`**：返回 `NoParallelAttention` 实例
- **`ring.py`**：返回 `RingParallelAttention` 实例
- **`ulysses.py`**：返回 `UlyssesParallelAttention` 实例
- **`distributed/parallel_state.py`**：获取 SP 进程组和 world size
- **`forward_context.py`**：获取前向上下文中的配置
- **`layer.py`**：在 `Attention.__init__` 中调用

## 总结

`factory.py` 是并行策略选择的单一入口点。它封装了复杂的条件判断逻辑（配置检查、进程组可用性检测、world size 验证），确保在任何环境下都能安全地返回一个可用的策略，即使降级到无并行。
