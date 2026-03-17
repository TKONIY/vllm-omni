# `version.py` — 版本管理

## 文件概述

`version.py` 管理 vllm-omni 的版本信息。版本号通过 `setuptools_scm` 从 git 标签自动生成，写入 `_version.py` 文件。

## 关键代码解析

```python
try:
    from ._version import __version__, __version_tuple__
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import version from _version.py: {e}\n"
        "This typically happens in development mode before building.\n"
        "Using fallback version 'dev'.",
        RuntimeWarning,
        stacklevel=2,
    )
    __version__ = "dev"
    __version_tuple__ = (0, 0, "dev")
```

- 正常情况下从构建生成的 `_version.py` 导入版本信息
- 开发模式下（未构建）使用 `"dev"` 作为回退版本
- 使用 `RuntimeWarning` 提醒用户

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `__version__` | 字符串 | 版本号，如 `"0.1.0"` |
| `__version_tuple__` | 元组 | 版本元组，如 `(0, 1, 0)` |

## 与其他模块的关系

- 被 `__init__.py` 导入并导出为包级别公共 API
- 依赖 `setuptools_scm` 的构建产物 `_version.py`

## 总结

标准的版本管理模块，支持自动版本号生成和开发模式回退。
