# `base.py` — 卸载策略与后端基类

## 文件概述

`base.py` 定义了 CPU 卸载系统的基础抽象，包括卸载策略枚举、配置数据类和后端抽象基类。

## 关键代码解析

### 1. 卸载策略枚举

```python
class OffloadStrategy(Enum):
    NONE = "none"
    MODEL_LEVEL = "model_level"   # DiT 与编码器之间的顺序卸载
    LAYER_WISE = "layer_wise"     # 块级卸载
```

支持三种策略：
- **NONE**：不卸载。
- **MODEL_LEVEL**：模型级顺序卸载，DiT 和编码器互斥占用 GPU。
- **LAYER_WISE**：层级卸载，transformer 的各个 block 滑动窗口式地加载到 GPU。

### 2. 卸载配置

```python
@dataclass
class OffloadConfig:
    strategy: OffloadStrategy
    pin_cpu_memory: bool = True

    @classmethod
    def from_od_config(cls, od_config: OmniDiffusionConfig) -> "OffloadConfig":
        enable_cpu_offload = getattr(od_config, "enable_cpu_offload", False)
        enable_layerwise_offload = getattr(od_config, "enable_layerwise_offload", False)

        # 互斥：layer_wise 优先级高于 model_level
        if enable_layerwise_offload:
            strategy = OffloadStrategy.LAYER_WISE
        elif enable_cpu_offload:
            strategy = OffloadStrategy.MODEL_LEVEL
        else:
            strategy = OffloadStrategy.NONE
```

从 `OmniDiffusionConfig` 中提取配置，当两种卸载同时启用时，层级卸载优先。`pin_cpu_memory` 控制是否使用页锁定内存加速 CPU-GPU 传输。

### 3. 后端基类

```python
class OffloadBackend(ABC):
    def __init__(self, config: OffloadConfig, device: torch.device):
        self.config = config
        self.device = device
        self.enabled = False

    @abstractmethod
    def enable(self, pipeline: nn.Module) -> None:
        """启用卸载：发现模块、移动到设备、注册 hook"""

    @abstractmethod
    def disable(self) -> None:
        """禁用卸载：移除 hook，但不移动模块"""
```

后端基类定义了统一的生命周期接口：`enable()` 启用卸载，`disable()` 清理资源。注意 `disable()` 不负责将模块移回原始设备。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OffloadStrategy` | 枚举 | 卸载策略：NONE / MODEL_LEVEL / LAYER_WISE |
| `OffloadConfig` | 数据类 | 卸载配置，包含策略和内存固定选项 |
| `OffloadConfig.from_od_config` | 类方法 | 从 OmniDiffusionConfig 提取并验证配置 |
| `OffloadBackend` | 抽象类 | 卸载后端基类，定义 enable/disable 接口 |

## 与其他模块的关系

- **`layerwise_backend.py`**：`LayerWiseOffloadBackend` 继承此基类。
- **`sequential_backend.py`**：`ModelLevelOffloadBackend` 继承此基类。
- **`__init__.py`**：`get_offload_backend` 使用 `OffloadConfig` 和 `OffloadStrategy` 决策。
- **`OmniDiffusionConfig`**：配置来源。

## 总结

`base.py` 为卸载系统建立了清晰的抽象层次：策略枚举描述"做什么"，配置数据类封装"怎么配"，后端基类规范"如何做"。两种具体策略通过互斥优先级机制处理冲突。
