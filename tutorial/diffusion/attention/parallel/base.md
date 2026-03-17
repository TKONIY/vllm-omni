# `base.py` — 并行注意力策略基类定义

## 文件概述

`base.py` 定义了并行注意力策略的核心接口 `ParallelAttentionStrategy`（Protocol）、上下文数据类 `ParallelAttentionContext`，以及默认的无并行策略 `NoParallelAttention`。

## 关键代码解析

### 1. ParallelAttentionContext — 策略上下文

```python
@dataclass(frozen=True, slots=True)
class ParallelAttentionContext:
    """Opaque per-forward context returned by a parallel strategy."""
    name: str
```

不可变的数据类，由 `pre_attention` 创建、在 `post_attention` 中使用。策略子类可以扩展此类以存储需要在后处理阶段使用的信息（如进程组、切片信息等）。

### 2. ParallelAttentionStrategy — 策略协议

```python
class ParallelAttentionStrategy(Protocol):
    """Pluggable strategy for parallel attention communication/resharding."""

    @property
    def enabled(self) -> bool: ...

    @property
    def name(self) -> str: ...

    def pre_attention(
        self, query, key, value, attn_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AttentionMetadata | None, ParallelAttentionContext | None]:
        """Runs before the attention kernel."""

    def post_attention(
        self, attn_output, ctx,
    ) -> torch.Tensor:
        """Runs after the attention kernel."""
```

使用 Python Protocol 而非抽象基类，实现结构化类型（duck typing），任何实现了这些方法的类都可以作为策略使用。接口分为：
- `enabled`：是否启用并行
- `pre_attention`：注意力计算前的通信/重分片
- `post_attention`：注意力计算后的逆向通信

### 3. NoParallelAttention — 默认无并行策略

```python
class NoParallelAttention:
    """Default strategy: do nothing (single device / no SP)."""

    @property
    def enabled(self) -> bool:
        return False

    def pre_attention(self, query, key, value, attn_metadata):
        return query, key, value, attn_metadata, None

    def post_attention(self, attn_output, ctx):
        return attn_output
```

直通实现：`pre_attention` 原样返回输入，`post_attention` 原样返回输出。用于单设备或不需要序列并行的场景。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ParallelAttentionStrategy` | Protocol | 并行策略的接口定义 |
| `ParallelAttentionContext` | 数据类 | `pre_attention` 创建的不可变上下文 |
| `NoParallelAttention` | 类 | 默认无操作策略（直通） |

## 与其他模块的关系

- **`ring.py`**：`RingParallelAttention` 实现 `ParallelAttentionStrategy`
- **`ulysses.py`**：`UlyssesParallelAttention` 实现 `ParallelAttentionStrategy`
- **`factory.py`**：在无法使用并行时返回 `NoParallelAttention`
- **`layer.py`**：`Attention` 使用策略的 `pre_attention`/`post_attention` 方法

## 总结

`base.py` 通过 Protocol 定义了并行注意力策略的统一接口，使用不可变数据类传递前后处理之间的上下文，并提供了零开销的默认直通策略。这种设计使得 `Attention` 类无需关心具体的并行实现细节。
