# `base.py` — Hook 基础机制

## 文件概述

`base.py` 实现了模型前向传播拦截（Hook）的基础框架。该机制允许在不修改模型代码的情况下，通过注册 Hook 来拦截和修改模块的 `forward` 调用。核心组件包括 `ModelHook`（Hook 基类）、`HookRegistry`（Hook 管理器）和状态管理类。

## 关键代码解析

### ModelHook — Hook 基类

```python
class ModelHook:
    def initialize_hook(self, module: nn.Module) -> nn.Module:
        """Hook 注册时的初始化回调"""
        return module

    def pre_forward(self, module, *args, **kwargs) -> tuple[tuple, dict]:
        """forward 前拦截，可修改输入"""
        return args, kwargs

    def post_forward(self, module, output) -> Any:
        """forward 后拦截，可修改输出"""
        return output

    def new_forward(self, module, *args, **kwargs) -> Any:
        """完全替代 forward 逻辑"""
        args, kwargs = self.pre_forward(module, *args, **kwargs)
        output = module._omni_original_forward(*args, **kwargs)
        return self.post_forward(module, output)
```

Hook 提供三个拦截点：`pre_forward`（输入变换）、`post_forward`（输出变换）和 `new_forward`（完全替代）。

### HookRegistry — Hook 注册管理器

```python
class HookRegistry:
    @classmethod
    def get_or_create(cls, module: nn.Module) -> HookRegistry:
        registry = getattr(module, "_hook_registry", None)
        if registry is None:
            registry = cls(module)
            module._hook_registry = registry
            # 保存原始 forward 并替换为 wrapped forward
            module._omni_original_forward = module.forward
            module.forward = _WrappedForward(module)
        return registry
```

首次为模块创建 Registry 时，保存原始 `forward` 为 `_omni_original_forward`，替换为 `_WrappedForward` 代理。

### 多 Hook 调度

```python
def dispatch(self, *args, **kwargs):
    if len(self._hooks) == 1:
        hook = next(iter(self._hooks.values()))
        return hook.new_forward(self.module, *args, **kwargs)

    # 多 Hook：按名称排序调度
    sorted_hooks = sorted(self._hooks.items(), key=lambda x: x[0])
    for _, hook in sorted_hooks:
        args, kwargs = hook.pre_forward(self.module, *args, **kwargs)
    output = self.module._omni_original_forward(*args, **kwargs)
    for _, hook in reversed(sorted_hooks):
        output = hook.post_forward(self.module, output)
    return output
```

单 Hook 走快速路径（调用 `new_forward`）；多 Hook 按名称字母序链式调用 `pre_forward`，然后逆序调用 `post_forward`。

### 状态管理

```python
class BaseState:
    def reset(self) -> None: pass

class StateManager:
    def __init__(self, state_cls):
        self._states: dict[str, BaseState] = {}
        self._context: str = "default"

    def get_state(self) -> BaseState:
        if self._context not in self._states:
            self._states[self._context] = self._state_cls()
        return self._states[self._context]
```

`StateManager` 支持按上下文（context）管理 Hook 状态，适用于需要在不同推理场景中维护独立状态的 Hook。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ModelHook` | 类 | Hook 基类，提供 pre/post/new_forward 拦截点 |
| `HookRegistry` | 类 | 模块级 Hook 管理器，负责注册、移除和调度 |
| `_WrappedForward` | dataclass | forward 代理，将调用转发到 HookRegistry |
| `BaseState` | 类 | Hook 状态基类 |
| `StateManager` | 类 | 按上下文管理 Hook 状态 |

## 与其他模块的关系

- 被 `hooks/sequence_parallel.py` 使用，`SequenceParallelSplitHook` 和 `SequenceParallelGatherHook` 继承 `ModelHook`
- 被 `registry.py` 间接使用（通过 `apply_sequence_parallel`）
- Hook 框架是非侵入式模型修改的基础设施，可用于序列并行、缓存加速等多种场景

## 总结

`base.py` 实现了一个灵活的 Hook 框架，通过替换模块的 `forward` 方法来拦截前向传播。它支持单 Hook 快速路径和多 Hook 链式调度，是序列并行等高级功能的基础设施。使用 `_omni_original_forward` 命名避免与 cache-dit 的 `_original_forward` 冲突。
