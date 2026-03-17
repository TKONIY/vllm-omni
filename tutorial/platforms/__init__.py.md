# `__init__.py` -- 平台自动检测与懒加载入口

## 文件概述

`platforms/__init__.py` 是整个平台抽象层的入口文件，承担两个核心职责：

1. **硬件平台自动检测**：在运行时自动探测当前环境可用的硬件加速器（CUDA、ROCm、NPU、XPU），并激活对应的平台插件。
2. **懒加载单例**：通过模块级 `__getattr__` 实现 `current_omni_platform` 的懒初始化，确保平台实例在首次访问时才创建。

## 关键代码解析

### 1. 内置平台探测函数

每种硬件都有对应的探测函数，通过调用硬件专属 SDK 检测设备是否可用：

```python
def cuda_omni_platform_plugin() -> str | None:
    """Check if CUDA OmniPlatform should be activated."""
    is_cuda = False
    try:
        from vllm.utils.import_utils import import_pynvml
        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() > 0:
                is_cuda = True
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("CUDA OmniPlatform is not available because: %s", str(e))
    return "vllm_omni.platforms.cuda.platform.CudaOmniPlatform" if is_cuda else None
```

四个探测函数的检测方式对比：

| 平台 | 检测方法 | 返回的全限定类名 |
|------|---------|----------------|
| CUDA | pynvml 检测 GPU 数量 | `CudaOmniPlatform` |
| ROCm | amdsmi 检测处理器句柄 | `RocmOmniPlatform` |
| NPU | `torch.npu.is_available()` | `NPUOmniPlatform` |
| XPU | `torch.xpu.is_available()` + 通信后端检测 | `XPUOmniPlatform` |

XPU 的探测逻辑额外检测分布式通信后端：

```python
def xpu_omni_platform_plugin() -> str | None:
    try:
        import torch
        if supports_xccl():
            dist_backend = "xccl"
        else:
            dist_backend = "ccl"
            import oneccl_bindings_for_pytorch  # noqa: F401
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            is_xpu = True
            XPUOmniPlatform.dist_backend = dist_backend
    except Exception as e:
        ...
```

### 2. 平台解析与互斥验证

`resolve_current_omni_platform_cls_qualname()` 函数执行完整的平台解析流程：

```python
def resolve_current_omni_platform_cls_qualname() -> str:
    platform_plugins = load_omni_plugins_by_group(OMNI_PLATFORM_PLUGINS_GROUP)
    activated_plugins = []
    for name, func in chain(builtin_omni_platform_plugins.items(), platform_plugins.items()):
        try:
            platform_cls_qualname = func()
            if platform_cls_qualname is not None:
                activated_plugins.append(name)
        except Exception:
            pass
    # ... 互斥检查：最多只能有一个平台被激活
```

关键设计点：
- **外部插件优先**：若存在第三方 OmniPlatform 插件且被激活，优先使用外部插件。
- **互斥保证**：同一时间只允许一个平台激活，多平台激活时抛出 `RuntimeError`。
- **回退机制**：无平台可用时，返回 `UnspecifiedOmniPlatform`。

### 3. 懒加载单例模式

通过重写模块的 `__getattr__` 和 `__setattr__` 实现懒初始化：

```python
_current_omni_platform = None

def __getattr__(name: str):
    if name == "current_omni_platform":
        global _current_omni_platform
        if _current_omni_platform is None:
            platform_cls_qualname = resolve_current_omni_platform_cls_qualname()
            _current_omni_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_omni_platform
```

`_init_trace` 记录初始化调用栈，方便调试平台初始化时序问题。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `cuda_omni_platform_plugin()` | 函数 | 检测 CUDA 平台可用性 |
| `rocm_omni_platform_plugin()` | 函数 | 检测 ROCm 平台可用性 |
| `npu_omni_platform_plugin()` | 函数 | 检测 NPU 平台可用性 |
| `xpu_omni_platform_plugin()` | 函数 | 检测 XPU 平台可用性 |
| `resolve_current_omni_platform_cls_qualname()` | 函数 | 解析当前应激活的平台类名 |
| `current_omni_platform` | 模块属性 | 当前平台单例（懒加载） |
| `builtin_omni_platform_plugins` | 字典 | 内置平台探测函数注册表 |

## 与其他模块的关系

- **上游依赖**：`vllm.utils.import_utils`（类加载）、`vllm_omni.plugins`（插件系统）
- **下游消费**：项目中所有需要获取当前硬件平台信息的模块均通过 `from vllm_omni.platforms import current_omni_platform` 访问
- **内部关系**：根据探测结果实例化 `cuda/platform.py`、`rocm/platform.py`、`npu/platform.py` 或 `xpu/platform.py` 中的具体平台类

## 总结

该文件实现了 vllm-omni 的硬件平台自动发现机制。通过插件化的探测函数和懒加载单例模式，上层代码只需引用 `current_omni_platform` 即可获得对应硬件的平台实例，实现了硬件无关的统一编程接口。设计上支持外部平台插件扩展，并通过互斥检查保证运行时只有一个平台实例处于活跃状态。
