# `base.py` -- 调度器抽象基类

## 文件概述

`base.py` 定义了所有扩散模型调度器（Scheduler）的抽象基类 `BaseScheduler`。它规定了子类必须实现的接口，包括设置时间步、缩放模型输入和调整 shift 参数。

**文件路径**: `vllm_omni/diffusion/models/schedulers/base.py`

## 关键代码解析

### BaseScheduler 抽象基类

```python
class BaseScheduler(ABC):
    timesteps: torch.Tensor
    order: int
    num_train_timesteps: int

    def __init__(self):
        required_attrs = ["timesteps", "order", "num_train_timesteps"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclass {self.__class__.__name__} must define `{attr}` "
                    f"before calling super().__init__()"
                )
```

构造函数进行属性检查，确保子类在调用 `super().__init__()` 之前已经初始化了必要属性。这是一种**防御性编程**模式，避免子类遗漏关键属性。

### 抽象方法

```python
@abstractmethod
def set_shift(self, shift: float) -> None: ...

@abstractmethod
def set_timesteps(self, *args, **kwargs) -> None: ...

@abstractmethod
def scale_model_input(self, sample: torch.Tensor, timestep: int | None = None) -> torch.Tensor: ...
```

- `set_shift`: 设置噪声调度的 shift 参数（不同分辨率需要不同 shift 值）
- `set_timesteps`: 设置推理时间步序列
- `scale_model_input`: 缩放模型输入（部分调度器需要）

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `BaseScheduler` | 抽象基类 | 所有自定义调度器的基类，定义必须实现的接口 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被继承 | `scheduling_flow_unipc_multistep.py` | `FlowUniPCMultistepScheduler` 继承此基类 |
| 参考来源 | FastVideo / diffusers | 改编自开源项目 |

## 总结

`BaseScheduler` 通过抽象基类模式为自定义调度器提供统一接口规范。其构造函数中的属性检查机制确保子类正确初始化关键状态（`timesteps`、`order`、`num_train_timesteps`），有效避免运行时错误。
