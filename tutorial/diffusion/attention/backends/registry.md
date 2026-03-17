# `registry.py` — 扩散模型注意力后端注册表

## 文件概述

`registry.py` 提供了一个基于枚举的后端注册表系统（`DiffusionAttentionBackendEnum`），用于管理所有支持的扩散模型注意力后端。它支持运行时覆盖已注册后端的实现类，使得平台可以替换或扩展默认后端。

## 关键代码解析

### 1. 自定义枚举元类

```python
class _DiffusionBackendEnumMeta(EnumMeta):
    def __getitem__(cls, name: str) -> "DiffusionAttentionBackendEnum":
        try:
            return super().__getitem__(name)
        except KeyError:
            members = list(cls.__members__.keys())
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown diffusion attention backend: '{name}'. Valid options are: {valid_backends}"
            ) from None
```

通过自定义元类，在访问不存在的后端名称时提供更友好的错误信息，列出所有可用的后端选项。

### 2. 后端枚举定义

```python
class DiffusionAttentionBackendEnum(Enum, metaclass=_DiffusionBackendEnumMeta):
    FLASH_ATTN = "vllm_omni.diffusion.attention.backends.flash_attn.FlashAttentionBackend"
    TORCH_SDPA = "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"
    SAGE_ATTN = "vllm_omni.diffusion.attention.backends.sage_attn.SageAttentionBackend"
```

每个枚举值是对应后端类的完全限定路径字符串。当前支持三种后端：
- `FLASH_ATTN`：Flash Attention（性能最优）
- `TORCH_SDPA`：PyTorch SDPA（通用兼容）
- `SAGE_ATTN`：Sage Attention（高效近似）

### 3. 运行时覆盖机制

```python
_DIFFUSION_ATTN_OVERRIDES: dict[DiffusionAttentionBackendEnum, str] = {}

def register_diffusion_backend(
    backend: DiffusionAttentionBackendEnum,
    class_path: str | None = None,
) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        _DIFFUSION_ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _DIFFUSION_ATTN_OVERRIDES[backend] = class_path
        return lambda x: x

    return decorator
```

覆盖机制支持两种用法：
1. **装饰器模式**：`@register_diffusion_backend(DiffusionAttentionBackendEnum.FLASH_ATTN)`
2. **直接注册**：`register_diffusion_backend(backend, "my.module.MyClass")`

### 4. 获取后端类（尊重覆盖）

```python
def get_path(self, include_classname: bool = True) -> str:
    path = _DIFFUSION_ATTN_OVERRIDES.get(self, self.value)
    if not path:
        raise ValueError(...)
    return path

def get_class(self) -> "type[AttentionBackend]":
    return resolve_obj_by_qualname(self.get_path())
```

`get_path()` 优先返回覆盖路径，否则返回枚举默认值。`get_class()` 通过 `resolve_obj_by_qualname` 动态加载后端类。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionAttentionBackendEnum` | 枚举 | 所有支持的注意力后端枚举 |
| `DiffusionAttentionBackendEnum.get_path` | 方法 | 获取后端类路径（尊重覆盖） |
| `DiffusionAttentionBackendEnum.get_class` | 方法 | 动态加载后端类 |
| `DiffusionAttentionBackendEnum.is_overridden` | 方法 | 检查后端是否被覆盖 |
| `register_diffusion_backend` | 函数 | 注册或覆盖后端实现 |
| `_DiffusionBackendEnumMeta` | 元类 | 提供友好的错误信息 |

## 与其他模块的关系

- **`abstract.py`**：注册的后端类必须是 `AttentionBackend` 的子类
- **`vllm_omni.platforms`**：平台层使用此注册表选择默认后端
- **`selector.py`**：通过平台层间接使用注册表

## 总结

`registry.py` 实现了一个灵活的后端注册表系统。通过枚举+覆盖字典的设计，既保证了类型安全和可发现性，又支持运行时动态替换后端实现。这种设计对于支持多硬件平台（如昇腾 NPU 需要替换默认的 Flash Attention 实现）尤为重要。
