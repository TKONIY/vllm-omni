# `__init__.py` — executor 包入口

## 文件概述

`executor/__init__.py` 是执行器子包的入口文件，内容为空。该文件将 `executor/` 目录标记为 Python 包，使得 `abstract.py` 和 `multiproc_executor.py` 可以被正确导入。

## 核心类/函数

无。

## 与其他模块的关系

- 使得 `DiffusionExecutor` 和 `MultiprocDiffusionExecutor` 可以通过 `vllm_omni.diffusion.executor` 路径导入。

## 总结

标准的 Python 包初始化文件，不包含业务逻辑。
