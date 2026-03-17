# `sequence_parallel.py` — 序列并行 Hooks

## 文件概述

`sequence_parallel.py` 实现了基于 Hook 机制的序列并行（Sequence Parallelism, SP）支持。它改编自 HuggingFace diffusers 的 Context Parallelism（CP），通过非侵入式的 Hook 注册实现输入分片和输出聚合，无需修改模型的 `forward()` 方法。支持 Ulysses、Ring 及混合并行模式。

## 关键代码解析

### ModuleForwardMetadata — 参数位置缓存

```python
@dataclass
class ModuleForwardMetadata:
    cached_parameter_indices: dict[str, int] | None = None
    _cls: type | None = None

    def _get_parameter_from_args_kwargs(self, identifier, args, kwargs):
        # 先检查 kwargs，再通过缓存的 forward 签名映射到 args 位置
        if identifier in kwargs:
            return kwargs[identifier], True, None
        # 首次调用时通过 inspect.signature 构建参数名 -> 位置索引缓存
```

缓存模块 `forward` 方法的参数签名，使得后续通过参数名高效定位到 args/kwargs 中的位置。

### SequenceParallelSplitHook — 输入分片

```python
class SequenceParallelSplitHook(ModelHook):
    def pre_forward(self, module, *args, **kwargs):
        for name, spm in self.metadata.items():
            if isinstance(spm, SequenceParallelInput) and spm.split_output:
                continue  # 输出分片在 post_forward 处理
            input_val = self._get_parameter(name, args, kwargs)
            input_val = self._prepare_sp_input(input_val, spm, args, kwargs)
            # 更新 args/kwargs
        return args, kwargs

    def post_forward(self, module, output):
        # 处理 split_output=True 的条目（分片输出而非输入）
        # 更新 _sp_shard_depth
```

支持两种分片模式：
- **SequenceParallelInput**：全序列分片，可选 `auto_pad`（自动补齐到可整除长度）
- **SequenceParallelPartialInput**：部分分片（保留文本部分，仅分片图像部分）

### SequenceParallelGatherHook — 输出聚合

```python
class SequenceParallelGatherHook(ModelHook):
    def post_forward(self, module, output):
        for i, spm in enumerate(self.metadata):
            gathered = sp_gather(output[i], spm.gather_dim, validate=False)
            # 移除 auto_pad 添加的 padding
            if original_seq_len is not None and gathered.size(spm.gather_dim) > original_seq_len:
                gathered = gathered.narrow(spm.gather_dim, 0, original_seq_len)
            output[i] = gathered
        # 递减 _sp_shard_depth
```

聚合后自动检查并移除 padding，确保输出与原始序列长度一致。

### auto_pad — 自动补齐

```python
def _shard_with_auto_pad(self, x, dim):
    seq_len = x.size(dim)
    remainder = seq_len % world_size
    if remainder == 0:
        return sp_shard(x, dim)
    # 检查注意力后端是否支持 attention_mask
    pad_size = world_size - remainder
    x_padded = torch.cat([x, padding], dim=dim)
    # 在 ForwardContext 中记录 padding 信息
    ctx.sp_padding_size = pad_size
    ctx.sp_original_seq_len = seq_len
    return x_padded.chunk(world_size, dim=dim)[rank]
```

当序列长度不能被 SP world_size 整除时，自动添加零值 padding 并记录到 `ForwardContext`，由 `GatherHook` 在聚合后移除。

### apply_sequence_parallel — 注册入口

```python
def apply_sequence_parallel(module, config, plan):
    for module_id, sp_model_plan in plan.items():
        submodule = _get_submodule_by_name(module, module_id)
        for m in submodule:
            if isinstance(sp_model_plan, dict):
                hook = SequenceParallelSplitHook(sp_model_plan, config)
            elif isinstance(sp_model_plan, SequenceParallelOutput):
                hook = SequenceParallelGatherHook(sp_model_plan, config)
            registry = HookRegistry.get_or_create(m)
            registry.register_hook(hook_name, hook)
```

根据模型的 `_sp_plan` 定义，为指定的子模块注册分片或聚合 Hook。支持通配符 `"*"` 匹配 `ModuleList` 的所有子模块。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `SequenceParallelSplitHook` | 类 | 输入分片 Hook，支持全分片和部分分片 |
| `SequenceParallelGatherHook` | 类 | 输出聚合 Hook，自动移除 padding |
| `ModuleForwardMetadata` | dataclass | 参数签名缓存，高效定位参数 |
| `apply_sequence_parallel` | 函数 | 根据 plan 注册 SP hooks |
| `remove_sequence_parallel` | 函数 | 移除 SP hooks |
| `enable_sequence_parallel_for_model` | 函数 | 便捷函数，自动从模型读取 `_sp_plan` 并应用 |
| `disable_sequence_parallel_for_model` | 函数 | 便捷函数，禁用序列并行 |

## 与其他模块的关系

- 继承 `hooks/base.py` 的 `ModelHook` 和 `HookRegistry`
- 使用 `distributed/sp_sharding.py` 的 `sp_shard` 和 `sp_gather` 进行实际的张量分片/聚合
- 使用 `distributed/sp_plan.py` 中的数据类（`SequenceParallelInput`, `SequenceParallelOutput` 等）
- 通过 `forward_context.py` 的 `_sp_shard_depth` 跟踪 SP 作用域
- 被 `registry.py` 的 `_apply_sequence_parallel_if_enabled` 调用

## 总结

`sequence_parallel.py` 是序列并行的核心实现，通过 Hook 机制在模型前向传播中自动进行输入分片和输出聚合。它支持全序列分片、文本-图像部分分片、自动 padding 等多种模式，并通过 `_sp_shard_depth` 精确控制 SP 的作用域。整体设计遵循 diffusers 的 Context Parallelism 架构，但针对 vLLM-Omni 的 Ulysses/Ring 混合并行做了适配。
