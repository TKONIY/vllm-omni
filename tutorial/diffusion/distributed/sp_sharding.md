# `sp_sharding.py` -- 序列并行分片工具

## 文件概述

`sp_sharding.py` 提供了序列并行的底层分片和聚合函数，以及用于调试的分片验证器。这些工具可以直接在模型 forward 方法中使用（半侵入式 SP），也可以被 SP Hook 系统内部调用。

## 关键代码解析

### sp_shard -- 序列分片

```python
def sp_shard(tensor, dim, validate=True):
    """将张量沿指定维度分片，当前 rank 获取对应的块。

    Args:
        tensor: 待分片张量
        dim: 分片维度
        validate: 是否验证可整除性

    Example:
        # 分片前: hidden_states.shape = (batch, seq_len, hidden_dim)
        hidden_states = sp_shard(hidden_states, dim=1)
        # 分片后: hidden_states.shape = (batch, seq_len/P, hidden_dim)
    """
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return tensor
    rank = get_sequence_parallel_rank()
    if validate and tensor.size(dim) % world_size != 0:
        raise ValueError(...)
    return tensor.chunk(world_size, dim=dim)[rank]
```

### sp_gather -- 序列聚合

```python
def sp_gather(tensor, dim, validate=True):
    """从所有序列并行 rank 聚合张量。

    Example:
        # 聚合前: output.shape = (batch, seq_len/P, hidden_dim)
        output = sp_gather(output, dim=1)
        # 聚合后: output.shape = (batch, seq_len, hidden_dim)
    """
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return tensor
    sp_group = get_sp_group()
    return sp_group.all_gather(tensor, dim=dim)
```

### sp_shard_with_padding -- 带填充的分片

```python
def sp_shard_with_padding(tensor, dim, pad_value=0.0):
    """分片张量，如果不可整除则自动填充。

    Returns:
        (sharded_tensor, padding_size)

    Example:
        sharded, pad_size = sp_shard_with_padding(hidden_states, dim=1)
        # ... 处理 ...
        output = sp_gather(output, dim=1)
        if pad_size > 0:
            output = output[..., :-pad_size]  # 移除填充
    """
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return tensor, 0

    size = tensor.size(dim)
    remainder = size % world_size
    if remainder == 0:
        return sp_shard(tensor, dim, validate=False), 0

    pad_size = world_size - remainder
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat([tensor, padding], dim=dim)
    return sp_shard(tensor, dim, validate=False), pad_size
```

### ShardingValidator -- 分片验证器

```python
@dataclass
class ShardingValidator:
    _sharded: set[str] = field(default_factory=set)
    _gathered: set[str] = field(default_factory=set)
    _enabled: bool = False

    @contextmanager
    def track(self):
        """启用跟踪的上下文管理器。"""
        self._enabled = True
        self.reset()
        try:
            yield
        finally:
            self._enabled = False

    def shard(self, tensor, name, dim, validate_divisible=True):
        """分片并跟踪操作。"""
        if self._enabled:
            if name in self._sharded:
                logger.warning(f"Tensor '{name}' sharded multiple times")
            self._sharded.add(name)
        return sp_shard(tensor, dim, validate=validate_divisible)

    def gather(self, tensor, name, dim):
        """聚合并跟踪操作。"""
        if self._enabled:
            if name not in self._sharded:
                logger.warning(f"Tensor '{name}' gathered without being sharded")
            self._gathered.add(name)
        return sp_gather(tensor, dim)

    def validate(self):
        """验证所有分片的张量都已被聚合。"""
        unmatched = self._sharded - self._gathered
        if unmatched:
            raise ValueError(f"These tensors were sharded but not gathered: {unmatched}")
```

使用示例：
```python
validator = get_sharding_validator()
with validator.track():
    hidden_states = validator.shard(hidden_states, "hidden_states", dim=1)
    # ... 模型计算 ...
    output = validator.gather(output, "hidden_states", dim=1)
validator.validate()  # 如果有遗漏的 shard/gather 会报错
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `sp_shard()` | 函数 | 沿指定维度分片张量 |
| `sp_gather()` | 函数 | 从所有 rank 聚合张量 |
| `sp_shard_with_padding()` | 函数 | 带自动填充的分片 |
| `ShardingValidator` | 类 | 分片/聚合操作的调试验证器 |
| `get_sharding_validator()` | 函数 | 获取全局验证器实例 |

## 与其他模块的关系

- **parallel_state.py**: 调用 `get_sequence_parallel_world_size()`, `get_sequence_parallel_rank()`, `get_sp_group()`
- **sp_plan.py**: `_sp_plan` 中声明的分片操作最终调用这些工具函数
- **模型 forward 方法**: 半侵入式 SP 直接调用 `sp_shard`/`sp_gather`

## 总结

`sp_sharding.py` 提供了三个层次的序列并行分片工具：
1. **基础工具** (`sp_shard`, `sp_gather`): 简洁的分片/聚合操作
2. **带填充工具** (`sp_shard_with_padding`): 处理不可整除序列长度
3. **调试验证器** (`ShardingValidator`): 帮助开发者确保分片和聚合正确配对

全局验证器实例是开发 SP 模型时的重要调试工具，能及时发现遗漏的 gather 或重复的 shard 等常见错误。
