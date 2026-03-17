# `__init__.py` — utils 包入口

## 文件概述

`utils/__init__.py` 是工具函数子包的入口文件，内容为空。该文件将 `utils/` 目录标记为 Python 包。

## 核心类/函数

无。

## 与其他模块的关系

- 使得 `hf_utils.py`、`network_utils.py`、`tf_utils.py` 可以通过 `vllm_omni.diffusion.utils` 路径导入。

## 总结

标准的 Python 包初始化文件，不包含业务逻辑。
