# `__init__.py` — layers 包入口

## 文件概述

`layers/__init__.py` 是层实现子包的入口文件，内容为空。该文件将 `layers/` 目录标记为 Python 包。

## 核心类/函数

无。

## 与其他模块的关系

- 使得 `adalayernorm.py`、`custom_op.py`、`rope.py` 等层实现可以通过 `vllm_omni.diffusion.layers` 路径导入。

## 总结

标准的 Python 包初始化文件，不包含业务逻辑。
