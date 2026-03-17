# `adalayernorm.py` — 自适应层归一化

## 文件概述

`adalayernorm.py` 实现了 `AdaLayerNorm`（自适应层归一化），这是扩散 Transformer 模型中的关键组件。它继承自 `CustomOp`，支持 CUDA、ROCm、NPU 和 XPU 等多平台调度。AdaLayerNorm 通过调制参数（shift、scale、gate）对 LayerNorm 的输出进行自适应调整。

## 关键代码解析

### AdaLayerNorm 数学公式

```
out = layernorm(x) * (1 + scale) + shift
```

输入 `mod_params` 被分成三个部分：shift、scale 和 gate，其中 gate 用于后续的门控机制。

### preprocess — 调制参数预处理

```python
def preprocess(self, mod_params, index=None):
    shift, scale, gate = mod_params.chunk(3, dim=-1)

    if index is not None:
        # CFG 场景：mod_params 的 batch 维度为 2*actual_batch
        actual_batch = shift.size(0) // 2
        # 根据 index 选择不同的调制参数
        shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
        scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
        gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
    else:
        shift_result = shift.unsqueeze(1)
        scale_result = scale.unsqueeze(1)
        gate_result = gate.unsqueeze(1)

    return shift_result, scale_result, gate_result
```

当 `index` 不为 None 时，支持 Classifier-Free Guidance（CFG）场景下的条件/无条件参数选择。

### 多平台 forward 实现

```python
def forward_native(self, x, mod_params, index=None):
    shift_result, scale_result, gate_result = self.preprocess(mod_params, index)
    return self.layernorm(x) * (1 + scale_result) + shift_result, gate_result

def forward_npu(self, x, mod_params, index=None):
    # 优先使用 mindiesd 的融合算子 layernorm_scale_shift
    # 回退到 torch_npu.npu_layer_norm_eval
```

- `forward_native`：通用 PyTorch 实现
- `forward_cuda`/`forward_hip`/`forward_xpu`：均委托给 `forward_native`
- `forward_npu`：优先使用 MindIE 融合算子，回退到 torch_npu

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `AdaLayerNorm` | 类 | 自适应层归一化，支持 CFG 条件分支 |
| `preprocess` | 方法 | 从 `mod_params` 中提取 shift/scale/gate |
| `forward_native` | 方法 | 通用 PyTorch 实现 |
| `forward_npu` | 方法 | NPU 平台实现，支持 MindIE 融合算子 |

## 与其他模块的关系

- 继承 `layers/custom_op.py` 的 `CustomOp` 基类
- 被扩散 Transformer 模型的注意力块使用（如 Qwen Image、Wan 等模型的 DiT Block）

## 总结

`AdaLayerNorm` 是扩散 Transformer 的核心归一化组件，通过学习到的调制参数自适应地调整归一化输出。它支持 CFG 场景下的条件/无条件分支选择，并通过 `CustomOp` 机制实现了多硬件平台的适配。
