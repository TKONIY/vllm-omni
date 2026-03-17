# `__init__.py` — 包初始化文件

## 文件概述

`__init__.py` 是 `vllm_omni/worker/` 包的初始化文件。当前为空文件（仅含一个空行），不包含任何导入或导出逻辑。

## 说明

该文件的存在使 `worker/` 目录被 Python 识别为一个包。其他模块通过显式路径导入所需的类，例如：

```python
from vllm_omni.worker.gpu_ar_worker import GPUARWorker
from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker
```

由于各 Worker 和 ModelRunner 通常由引擎根据阶段配置动态加载，不需要在 `__init__.py` 中预先导入。

## 与其他模块的关系

- 作为包标识符，使 `vllm_omni.worker` 命名空间下的所有子模块可被外部引用。

## 总结

该文件为标准的 Python 包初始化文件，无实质逻辑。
