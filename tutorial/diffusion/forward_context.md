# `forward_context.py` — 前向传播上下文管理

## 文件概述

`forward_context.py` 实现了一个全局前向传播上下文管理机制，用于在模型前向传播期间存储和传递配置信息。这使得 Attention 层、序列并行 hooks 等组件可以在不修改函数签名的情况下获取必要的配置（如注意力元数据、并行配置等）。

## 关键代码解析

### ForwardContext 数据类

```python
@dataclass
class ForwardContext:
    vllm_config: VllmConfig | None = None
    omni_diffusion_config: OmniDiffusionConfig | None = None
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None = None
    split_text_embed_in_sp: bool = False
    sp_padding_size: int = 0
    sp_original_seq_len: int | None = None
    sp_plan_hooks_applied: bool = False
    _sp_shard_depth: int = 0
```

上下文中存储了：
- **vllm_config**：vLLM 全局配置
- **omni_diffusion_config**：扩散模型配置
- **attn_metadata**：注意力元数据，供 Attention 层使用
- **sp_\* 字段**：序列并行相关状态（padding 大小、分片深度等）

### sp_active 属性 — 序列并行状态判断

```python
@property
def sp_active(self) -> bool:
    if self.sp_plan_hooks_applied:
        return self._sp_shard_depth > 0
    sp_size = self.omni_diffusion_config.parallel_config.sequence_parallel_size
    return sp_size is not None and sp_size > 1
```

`sp_active` 通过 `_sp_shard_depth` 跟踪当前是否处于序列并行分片区域内，Attention 层据此决定是否启用并行通信。

### 上下文管理器

```python
@contextmanager
def set_forward_context(
    vllm_config=None, omni_diffusion_config=None,
    attn_metadata=None, split_text_embed_in_sp=False,
):
    forward_context = create_forward_context(...)
    with override_forward_context(forward_context):
        if vllm_config is None:
            yield
        else:
            with set_current_vllm_config(vllm_config):
                yield
```

`set_forward_context` 是主要的使用接口，通过 `with` 语句设置上下文，退出时自动恢复。它同时设置 vLLM 的全局配置（用于 CustomOp 调度）。

### 全局状态管理

```python
_forward_context: ForwardContext | None = None

def get_forward_context() -> ForwardContext:
    assert _forward_context is not None
    return _forward_context

def is_forward_context_available() -> bool:
    return _forward_context is not None
```

使用模块级全局变量存储当前上下文，提供安全的访问接口。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ForwardContext` | dataclass | 前向传播上下文，存储配置和并行状态 |
| `set_forward_context` | 上下文管理器 | 设置前向传播上下文的主要接口 |
| `override_forward_context` | 上下文管理器 | 临时覆盖当前上下文 |
| `get_forward_context` | 函数 | 获取当前上下文（不存在时抛异常） |
| `is_forward_context_available` | 函数 | 检查上下文是否已设置 |
| `create_forward_context` | 函数 | 创建新的上下文实例 |

## 与其他模块的关系

- 被 `worker/diffusion_model_runner.py` 在模型执行时调用 `set_forward_context` 设置上下文
- 被 `worker/diffusion_worker.py` 在模型加载和初始化时设置上下文
- 被 `hooks/sequence_parallel.py` 用于跟踪 `_sp_shard_depth`（分片/聚集深度）
- 被 Attention 层读取 `attn_metadata` 和 `sp_active` 来决定并行策略
- 被 `registry.py` 在应用序列并行后更新 `sp_plan_hooks_applied`

## 总结

`forward_context.py` 实现了一种基于全局变量和上下文管理器的依赖注入模式，使得模型前向传播中的各组件可以无侵入地获取配置信息和并行状态。它是 Attention 层与序列并行机制的关键桥梁，通过 `_sp_shard_depth` 精确跟踪序列并行的作用域。
