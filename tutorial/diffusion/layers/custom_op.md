# `custom_op.py` — 多平台算子调度基类

## 文件概述

`custom_op.py` 定义了 `CustomOp` 基类，为扩散模块中的自定义算子提供多硬件平台的自动调度机制。子类只需实现各平台的 `forward_*` 方法，`CustomOp` 会在初始化时根据当前平台自动选择正确的实现。

## 关键代码解析

### CustomOp — 平台调度基类

```python
class CustomOp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def dispatch_forward(self) -> Callable:
        if current_omni_platform.is_rocm():
            return self.forward_hip
        elif current_omni_platform.is_cuda():
            return self.forward_cuda
        elif current_omni_platform.is_npu():
            return self.forward_npu
        elif current_omni_platform.is_xpu():
            return self.forward_xpu
        else:
            return self.forward_native

    def forward(self, *args, **kwargs) -> Any:
        return self._forward_method(*args, **kwargs)
```

调度流程：
1. 初始化时通过 `dispatch_forward()` 检测当前平台
2. 将对应平台的 `forward_*` 方法缓存到 `_forward_method`
3. 运行时 `forward()` 直接调用缓存的方法，零开销调度

### 平台接口

```python
def forward_native(self, *args, **kwargs):
    """PyTorch 原生实现，可用于编译和测试"""
    raise NotImplementedError

def forward_cuda(self, *args, **kwargs):
    raise NotImplementedError

def forward_npu(self, *args, **kwargs):
    raise NotImplementedError

def forward_xpu(self, *args, **kwargs):
    raise NotImplementedError

def forward_hip(self, *args, **kwargs):
    # 默认假设 HIP 算子与 CUDA 兼容
    return self.forward_cuda(*args, **kwargs)
```

`forward_hip` 默认回退到 `forward_cuda`，因为 AMD ROCm 的 HIP 接口通常兼容 CUDA。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `CustomOp` | 类 | 多平台算子调度基类 |
| `dispatch_forward` | 方法 | 根据当前平台选择 forward 实现 |
| `forward_native` | 方法 | PyTorch 原生实现（子类需覆盖） |
| `forward_cuda` | 方法 | CUDA 平台实现（子类需覆盖） |
| `forward_npu` | 方法 | NPU 平台实现（子类需覆盖） |
| `forward_xpu` | 方法 | XPU 平台实现（子类需覆盖） |
| `forward_hip` | 方法 | ROCm 平台实现，默认回退到 CUDA |

## 与其他模块的关系

- 被 `layers/adalayernorm.py` 的 `AdaLayerNorm` 继承
- 被 `layers/rope.py` 的 `RotaryEmbedding` 继承
- 依赖 `vllm_omni.platforms` 的 `current_omni_platform` 进行平台检测

## 总结

`CustomOp` 通过简单的继承和方法覆盖模式，实现了算子的多平台适配。子类只需实现特定平台的 `forward_*` 方法，平台调度在初始化时一次性完成，运行时无额外开销。这种设计使得同一份算子代码可以透明地运行在 CUDA、ROCm、NPU 和 XPU 等不同硬件上。
