# `selector.py` — 扩散模型注意力后端选择器

## 文件概述

`selector.py` 提供了扩散模型注意力后端的选择接口。它将后端选择逻辑委托给平台层（`vllm_omni.platforms`），类似于 vLLM 处理注意力后端选择的方式。支持通过环境变量 `DIFFUSION_ATTENTION_BACKEND` 覆盖默认选择。

## 关键代码解析

### 1. 动态加载后端类

```python
def _load_backend_cls(cls_path: str) -> type[AttentionBackend]:
    """从完全限定路径加载后端类。"""
    module_path, class_name = cls_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        backend_class = getattr(module, class_name)
        return backend_class
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in module: {e}")
```

通过字符串形式的完全限定类路径（如 `"vllm_omni.diffusion.attention.backends.flash_attn.FlashAttentionBackend"`）动态加载后端类，实现了后端的延迟导入和可扩展性。

### 2. 缓存化的后端获取

```python
@cache
def get_attn_backend(head_size: int) -> type[AttentionBackend]:
    from vllm_omni.platforms import current_omni_platform

    # 检查环境变量以支持用户覆盖
    selected_backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND")

    # 委托给平台层进行后端选择
    backend_cls_path = current_omni_platform.get_diffusion_attn_backend_cls(
        selected_backend=selected_backend,
        head_size=head_size,
    )

    return _load_backend_cls(backend_cls_path)
```

核心设计要点：
- 使用 `@cache` 装饰器，同一 `head_size` 只会执行一次后端选择
- 支持环境变量 `DIFFUSION_ATTENTION_BACKEND` 进行用户级覆盖
- 将实际选择逻辑委托给 `current_omni_platform`，使得不同硬件平台（CUDA / ROCm / NPU / XPU）可以选择各自最优的后端

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_attn_backend` | 函数 | 获取当前平台的最优注意力后端类（带缓存） |
| `_load_backend_cls` | 函数 | 从完全限定路径动态加载后端类 |

## 与其他模块的关系

- **`backends/abstract.py`**：返回的类型为 `AttentionBackend` 的子类
- **`layer.py`**：在 `Attention.__init__` 中调用 `get_attn_backend()` 获取后端
- **`vllm_omni.platforms`**：委托给平台层决定具体使用哪个后端
- **`backends/registry.py`**：平台层内部使用注册表枚举来管理后端映射

## 总结

`selector.py` 是注意力后端选择的统一入口。它通过平台委托和环境变量覆盖的双重机制，既保证了自动选择最优后端的便利性，又提供了手动指定后端的灵活性。`@cache` 装饰器避免了重复的后端解析开销。
