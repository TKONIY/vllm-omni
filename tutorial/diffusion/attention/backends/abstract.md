# `abstract.py` — 注意力后端抽象基类定义

## 文件概述

`abstract.py` 定义了扩散模型注意力后端的抽象接口，包括 `AttentionBackend`（后端工厂）、`AttentionImpl`（后端实现）和 `AttentionMetadata`（注意力元数据）。所有具体的注意力后端（Flash Attention、SDPA、Sage 等）都必须继承这些基类。

## 关键代码解析

### 1. AttentionBackend — 后端工厂抽象类

```python
class AttentionBackend(ABC):
    """Abstract class for diffusion attention backends."""

    accept_output_buffer: bool = False

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes
```

`AttentionBackend` 是工厂模式的抽象基类，子类必须实现：
- `get_name()`：返回后端名称
- `get_impl_cls()`：返回对应的 `AttentionImpl` 实现类
- `get_metadata_cls()`：返回元数据类
- `get_supported_head_sizes()`：返回支持的 head size 列表

### 2. AttentionMetadata — 注意力元数据

```python
@dataclass
class AttentionMetadata:
    attn_mask: torch.Tensor | None = None
    joint_attn_mask: torch.Tensor | None = None
    joint_query: torch.Tensor | None = None
    joint_key: torch.Tensor | None = None
    joint_value: torch.Tensor | None = None
    joint_strategy: str = "front"
```

`AttentionMetadata` 用于传递注意力计算所需的额外信息：
- `attn_mask` / `joint_attn_mask`：注意力掩码
- `joint_query/key/value`：联合注意力张量（如文本+图像场景中的文本条件），在各进程间复制
- `joint_strategy`：联合张量的拼接策略，`"front"` 表示拼接到前面，`"rear"` 表示拼接到后面

### 3. AttentionImpl — 注意力实现抽象类

```python
class AttentionImpl(ABC, Generic[T]):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        """Dispatch to platform-specific forward implementation."""
        if current_omni_platform.is_rocm():
            return self.forward_hip(query, key, value, attn_metadata)
        elif current_omni_platform.is_cuda():
            return self.forward_cuda(query, key, value, attn_metadata)
        elif current_omni_platform.is_npu():
            return self.forward_npu(query, key, value, attn_metadata)
        elif current_omni_platform.is_xpu():
            return self.forward_xpu(query, key, value, attn_metadata)
```

`AttentionImpl` 的 `forward()` 方法实现了平台分派机制：
- 根据当前平台自动路由到 `forward_cuda()`、`forward_hip()`、`forward_npu()` 或 `forward_xpu()`
- 子类只需实现对应平台的方法即可
- HIP（ROCm）默认回退到 CUDA 实现

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `AttentionBackend` | 抽象类 | 注意力后端的工厂基类，定义后端能力查询接口 |
| `AttentionMetadata` | 数据类 | 注意力计算的元数据，包含掩码和联合张量信息 |
| `AttentionImpl` | 抽象类 | 注意力实现基类，提供平台分派的 forward 方法 |
| `AttentionImpl.forward` | 方法 | 根据平台类型自动路由到对应的 forward 实现 |
| `AttentionBackend.supports_head_size` | 方法 | 检查后端是否支持指定的 head size |

## 与其他模块的关系

- **`flash_attn.py`**：`FlashAttentionBackend` / `FlashAttentionImpl` 继承这些基类
- **`sdpa.py`**：`SDPABackend` / `SDPAImpl` 继承这些基类
- **`sage_attn.py`**：`SageAttentionBackend` / `SageAttentionImpl` 继承这些基类
- **`layer.py`**：使用 `AttentionMetadata` 传递元数据
- **`parallel/`**：并行策略操作 `AttentionMetadata` 中的联合张量

## 总结

`abstract.py` 是整个注意力后端体系的基石。它通过抽象基类定义了后端工厂和实现的统一接口，通过 `AttentionMetadata` 数据类统一了元数据传递方式，通过平台分派机制实现了跨硬件平台的透明支持。联合注意力（joint attention）相关字段的设计支持了文本+图像等多模态场景。
