# `__init__.py` — diffusion 模块入口

## 文件概述

`__init__.py` 是 `vllm_omni/diffusion/` 包的入口文件。该文件内容为空（仅包含一个空行），其作用是将 `diffusion/` 目录标记为 Python 包，使得其他模块可以通过 `import vllm_omni.diffusion` 来访问该包下的子模块。

## 关键代码解析

该文件不包含任何实质性代码，仅作为包初始化标记文件存在。

## 核心类/函数

无。

## 与其他模块的关系

- 作为 `vllm_omni.diffusion` 包的根标识，使得 `data.py`、`diffusion_engine.py`、`registry.py` 等模块可以被正确导入。
- 子包 `executor/`、`hooks/`、`layers/`、`utils/`、`worker/`、`profiler/` 等也通过各自的 `__init__.py` 暴露接口。

## 总结

该文件是标准的 Python 包初始化文件，不包含业务逻辑，仅用于声明 `diffusion/` 为一个可导入的 Python 包。
