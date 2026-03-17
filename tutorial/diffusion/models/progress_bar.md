# `progress_bar.py` -- 扩散流水线进度条混入类

## 文件概述

`progress_bar.py` 提供了一个 `ProgressBarMixin` 混入类，为扩散管线的去噪循环提供与 diffusers 兼容的进度条功能。在分布式环境中，进度条仅在 rank 0 进程上显示，避免多进程输出冲突。

**文件路径**: `vllm_omni/diffusion/models/progress_bar.py`

## 关键代码解析

### ProgressBarMixin 混入类

```python
class ProgressBarMixin:
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        config = dict(self._progress_bar_config)
        # 仅在 rank 0 上显示进度条
        if "disable" not in config:
            config["disable"] = not _is_rank_zero()
        if iterable is not None:
            return tqdm(iterable, **config)
        elif total is not None:
            return tqdm(total=total, **config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")
```

该方法返回一个 `tqdm` 进度条实例。两种使用方式：
- 传入 `iterable`：直接包装可迭代对象
- 传入 `total`：创建手动更新的进度条

### 分布式 rank 检测

```python
def _is_rank_zero() -> bool:
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0
```

在非分布式环境中默认返回 `True`，确保单 GPU 场景下正常显示进度条。

### 配置方法

```python
def set_progress_bar_config(self, **kwargs):
    self._progress_bar_config = kwargs
```

允许外部配置进度条参数（如 `desc`、`leave` 等 tqdm 参数）。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ProgressBarMixin` | Mixin 类 | 提供 `progress_bar()` 和 `set_progress_bar_config()` 方法 |
| `_is_rank_zero()` | 辅助函数 | 检测当前进程是否为 rank 0 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被混入 | 各 Pipeline 类 | Pipeline 继承此 Mixin 以获得进度条功能 |
| 依赖 | `torch.distributed` | 用于分布式环境检测 |
| 依赖 | `tqdm.auto` | 进度条的底层实现 |

## 总结

`ProgressBarMixin` 是一个轻量级工具类，为扩散管线的去噪循环提供进度显示功能。其核心设计考量是分布式友好——在多 GPU 场景下仅在 rank 0 显示进度条，避免终端输出混乱。Pipeline 类通过混入（Mixin）方式使用该功能，保持了低耦合的设计。
