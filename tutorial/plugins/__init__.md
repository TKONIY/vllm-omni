# `__init__.py` — 插件加载系统

## 文件概述

该文件实现了 vllm-omni 的插件发现和加载机制。基于 Python 的 `importlib.metadata.entry_points` 系统，支持通过 `pyproject.toml` 或 `setup.py` 注册插件，并在运行时自动发现和加载。

## 关键代码解析

### 插件组定义

```python
OMNI_DEFAULT_PLUGINS_GROUP = "vllm_omni.general_plugins"
OMNI_PLATFORM_PLUGINS_GROUP = "vllm_omni.platform_plugins"
```

定义了两个插件组：
- **general_plugins**: 通用插件，在所有进程中加载（process0、engine core、worker）
- **platform_plugins**: 平台插件，在 `current_omni_platform` 初始化时加载

### 按组加载插件

```python
def load_omni_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
    allowed_plugins = envs.VLLM_PLUGINS  # 可通过环境变量控制加载哪些插件

    discovered_plugins = entry_points(group=group)

    plugins: dict[str, Callable[[], Any]] = {}
    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            try:
                func = plugin.load()
                plugins[plugin.name] = func
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)
    return plugins
```

加载逻辑：
1. 通过 `entry_points(group=group)` 发现注册的插件
2. 如果设置了 `VLLM_PLUGINS` 环境变量，只加载允许列表中的插件
3. 调用 `plugin.load()` 加载插件（返回可调用对象）
4. 加载失败时记录异常但不中断

日志级别策略：对默认插件组使用 DEBUG 级别，对非默认组使用 INFO 级别。

### 通用插件加载

```python
omni_plugins_loaded = False

def load_omni_general_plugins() -> None:
    global omni_plugins_loaded
    if omni_plugins_loaded:
        return
    omni_plugins_loaded = True

    plugins = load_omni_plugins_by_group(group=OMNI_DEFAULT_PLUGINS_GROUP)
    for func in plugins.values():
        func()
```

通过全局标志 `omni_plugins_loaded` 确保每个进程只加载一次插件。加载后立即执行所有插件函数。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OMNI_DEFAULT_PLUGINS_GROUP` | 常量 | 通用插件组名称 |
| `OMNI_PLATFORM_PLUGINS_GROUP` | 常量 | 平台插件组名称 |
| `load_omni_plugins_by_group(group)` | 函数 | 按组发现和加载插件 |
| `load_omni_general_plugins()` | 函数 | 加载并执行通用插件（幂等） |
| `omni_plugins_loaded` | 全局变量 | 防止重复加载的标志 |

## 与其他模块的关系

- **使用 vLLM 环境变量**: 通过 `vllm.envs.VLLM_PLUGINS` 控制插件白名单。
- **被启动流程调用**: 在 vllm-omni 进程初始化时调用 `load_omni_general_plugins()`。
- **与 vLLM 插件系统并行**: 使用独立的 `vllm_omni.*` 插件组，不干扰 vLLM 原生插件。

## 总结

该文件实现了一个简洁的插件系统，基于 Python 标准的 entry_points 机制。核心特点是：支持环境变量控制加载范围、幂等加载（多次调用安全）、异常隔离（单个插件失败不影响其他插件）。插件开发者只需在 `pyproject.toml` 中注册 `vllm_omni.general_plugins` 或 `vllm_omni.platform_plugins` 入口点即可。
