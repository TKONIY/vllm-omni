# `logger.py` — 日志系统配置

## 文件概述

`logger.py` 负责将 vllm-omni 的日志系统桥接到 vLLM 的日志体系中，确保所有 `vllm_omni.*` 模块的日志输出都通过 vLLM 的根 logger 统一管理。

## 关键代码解析

### 日志层级桥接

```python
def _configure_vllm_omni_root_logger():
    vllm_root = logging.getLogger("vllm")
    vllm_omni_root = logging.getLogger("vllm_omni")
    vllm_omni_root.handlers = []
    vllm_omni_root.parent = vllm_root
    vllm_omni_root.propagate = True
    vllm_omni_root.setLevel(logging.NOTSET)
```

核心操作：
1. **清空 handlers**：避免重复输出
2. **设置 parent**：将 `vllm_omni` 的父 logger 设为 `vllm`，使日志沿 `vllm` 的 handler 链路输出
3. **启用 propagate**：日志消息向上传播
4. **level 设为 NOTSET**：不过滤任何级别，由 vLLM 的配置统一控制

### 自动执行

```python
_configure_vllm_omni_root_logger()
init_logger(__name__)
```

模块导入时自动执行配置，无需手动调用。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `_configure_vllm_omni_root_logger` | 函数 | 配置根 logger 的桥接关系 |

## 与其他模块的关系

- 被 `patch.py` 通过 `import vllm_omni.logger` 导入，确保日志配置在补丁之前完成
- 依赖 `vllm.logger.init_logger` 创建具名 logger
- 所有 vllm-omni 子模块的日志都通过此配置统一输出

## 总结

该文件通过简洁的 logger 层级设置，实现了 vllm-omni 与 vLLM 日志系统的无缝集成，用户无需额外配置即可获得统一的日志输出。
