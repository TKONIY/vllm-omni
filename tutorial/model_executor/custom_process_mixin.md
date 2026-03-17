# `custom_process_mixin.py` -- 阶段自定义前后处理 Mixin 基类

## 文件概述

`custom_process_mixin.py` 定义了 `CustomProcessMixin` 类，这是一个 Mixin（混入）基类，为 Omni 模型的各个推理阶段（stage）提供统一的前处理（preprocess）和后处理（postprocess）接口。通过该 Mixin，可以在运行时动态地为任意阶段注入自定义的处理逻辑。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/custom_process_mixin.py`

## 关键代码解析

### CustomProcessMixin 类

该类提供四个核心方法：两个 setter 用于注入自定义函数，两个默认实现作为未注入时的占位：

```python
class CustomProcessMixin:
    """
    Mixin class for all stages in the Omni model.
    """

    def set_custom_preprocess(self, preprocess_fn: Callable) -> None:
        """设置阶段的前处理函数"""
        self.preprocess = preprocess_fn

    def set_custom_postprocess(self, postprocess_fn: Callable) -> None:
        """设置阶段的后处理函数"""
        self.postprocess = postprocess_fn

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **input_dict: object
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """默认前处理：未实现时抛出 NotImplementedError"""
        raise NotImplementedError("Preprocess is not implemented for this stage.")

    def postprocess(self, model_output, **info_dict: object):
        """默认后处理：未实现时抛出 NotImplementedError"""
        raise NotImplementedError("Postprocess is not implemented for this stage.")
```

### 设计模式解析

该类采用了**策略模式**的变体：通过 `set_custom_preprocess` / `set_custom_postprocess` 方法，将实例方法替换为外部传入的函数。这意味着：

1. 调用 `set_custom_preprocess(fn)` 后，`self.preprocess` 不再是类上定义的默认方法，而是被替换为 `fn`
2. 如果没有调用 setter，调用 `preprocess` / `postprocess` 会抛出 `NotImplementedError`
3. 不同的阶段实例可以有完全不同的处理逻辑

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `CustomProcessMixin` | 类 | Mixin 基类，提供前后处理接口 |
| `set_custom_preprocess` | 方法 | 注入自定义前处理函数 |
| `set_custom_postprocess` | 方法 | 注入自定义后处理函数 |
| `preprocess` | 方法 | 前处理入口，接收 input_ids 和 input_embeds |
| `postprocess` | 方法 | 后处理入口，接收模型输出 |

## 与其他模块的关系

- **models/ 中的模型类**: 各个模型阶段（如 Thinker、Talker）可以混入此 Mixin，获得自定义处理能力
- **stage_input_processors/**: 该目录中定义的处理函数可以通过 setter 注入到对应阶段
- **engine/**: 引擎在初始化阶段时，根据 YAML 配置调用 setter 方法注入相应的处理逻辑

## 总结

`CustomProcessMixin` 是一个轻量但重要的 Mixin 基类，它通过运行时方法替换（而非继承重写）实现了灵活的前后处理注入机制。这种设计使得同一个模型类在不同阶段可以拥有完全不同的数据处理行为，是 vllm-omni 多阶段流水线架构的基础组件之一。
