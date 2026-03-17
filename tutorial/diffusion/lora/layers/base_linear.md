# `base_linear.py` — Diffusion LoRA 基础线性层

## 文件概述

`base_linear.py` 定义了 `DiffusionBaseLinearLayerWithLoRA`，这是所有扩散模型 LoRA 层的基类。它继承自 vLLM 的 `BaseLinearLayerWithLoRA`，但将 LoRA 的计算方式从 punica_wrapper（vLLM 的高效多 LoRA 批处理内核）替换为直接的 torch matmul 操作。这是因为扩散模型推理场景下通常只有单个 LoRA 处于激活状态，不需要 punica 的多 LoRA 批处理能力。

## 关键代码解析

### 1. 权重创建与属性转发

```python
def create_lora_weights(self, max_loras, lora_config, model_config=None):
    super().create_lora_weights(max_loras, lora_config, model_config)
    modules = object.__getattribute__(self, "_modules")
    base_layer = modules.get("base_layer") or object.__getattribute__(self, "__dict__").get("base_layer")
    object.__setattr__(self, "_diffusion_base_layer_ref", base_layer)
    n_slices = getattr(self, "n_slices", 1)
    self._diffusion_lora_active_slices = (False,) * int(n_slices)
```

创建 LoRA 权重后，额外保存一个对 `base_layer` 的直接引用到 `__dict__` 中，用于后续的属性转发。同时初始化 `_diffusion_lora_active_slices` 跟踪每个 slice 是否有活跃的 LoRA。

### 2. LoRA 激活状态跟踪

```python
def set_lora(self, index, lora_a, lora_b):
    super().set_lora(index, lora_a, lora_b)
    if isinstance(lora_a, list) or isinstance(lora_b, list):
        active_slices = []
        for a_i, b_i in zip(lora_a[:n_slices], lora_b[:n_slices]):
            active_slices.append(a_i is not None and b_i is not None)
        self._diffusion_lora_active_slices = tuple(active_slices)
    else:
        self._diffusion_lora_active_slices = (True,)
```

`set_lora` 在设置权重的同时记录哪些 slice 被激活。对于 packed 层（如 QKV），部分子投影可能没有 LoRA 权重，通过 `_diffusion_lora_active_slices` 可以快速跳过这些 slice。

### 3. 核心 apply 方法

```python
def apply(self, x, bias=None):
    output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

    # 快速路径：如果没有活跃的 LoRA，直接返回
    active_slices = getattr(self, "_diffusion_lora_active_slices", None)
    if active_slices is not None and not any(active_slices):
        return output

    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = output.reshape(-1, output.shape[-1])

    offset = 0
    for slice_idx, slice_size in enumerate(output_slices):
        if active_slices is not None and not active_slices[slice_idx]:
            offset += slice_size
            continue

        A = self.lora_a_stacked[slice_idx][0, 0, :, :]  # (rank, in_dim)
        B = self.lora_b_stacked[slice_idx][0, 0, :, :]  # (out_dim, rank)

        # LoRA 收缩与扩展：y += (x @ A^T) @ B^T
        delta = (x_flat @ A.t()) @ B.t()
        y_flat[:, offset:offset + slice_size] += delta
        offset += slice_size

    return y_flat.view(original_shape)
```

这是整个类的核心。计算过程分为两步：
1. **基础前向**：调用原始层的量化方法执行基础线性变换。
2. **LoRA 增量**：对每个活跃的 slice，执行 `x @ A^T @ B^T` 得到 LoRA 增量并累加到输出上。

这与 punica 内核的 `add_lora_linear()` 语义完全一致，但使用标准 PyTorch 矩阵乘法实现。

### 4. 属性转发机制

```python
def __getattr__(self, name):
    try:
        return super().__getattr__(name)
    except AttributeError as exc:
        base_layer = object.__getattribute__(self, "__dict__").get("_diffusion_base_layer_ref")
        if base_layer is not None:
            return getattr(base_layer, name)
        raise exc
```

扩散模型实现中经常直接访问线性层的属性（如 `QKVParallelLinear.num_heads`）。vLLM 的 LoRA 封装层默认不会转发这些属性，因此这里通过自定义 `__getattr__` 将未找到的属性查找委托给底层的 `base_layer`。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionBaseLinearLayerWithLoRA` | 类 | 扩散模型 LoRA 层基类 |
| `create_lora_weights` | 方法 | 创建 LoRA 权重缓冲区并初始化属性转发 |
| `set_lora` | 方法 | 设置 LoRA 权重并跟踪 slice 激活状态 |
| `reset_lora` | 方法 | 重置 LoRA 权重并清除激活状态 |
| `apply` | 方法 | 使用 torch matmul 执行 LoRA 计算（替代 punica） |
| `__getattr__` | 方法 | 将未匹配的属性查找转发到 base_layer |

## 与其他模块的关系

- **vLLM `BaseLinearLayerWithLoRA`**：直接继承，复用其权重管理、TP 切分等功能。
- **`column_parallel_linear.py`、`row_parallel_linear.py`、`replicated_linear.py`**：这些文件中的类都继承自本基类。
- **`../manager.py`**：通过 `set_lora` 和 `reset_lora` 方法与管理器交互。

## 总结

`DiffusionBaseLinearLayerWithLoRA` 是扩散 LoRA 层体系的基石。它通过用简单的 torch matmul 替换 punica_wrapper，使 LoRA 在扩散模型中无需依赖 vLLM 的专用 GPU 内核即可工作。同时通过 slice 级别的激活状态跟踪和属性转发机制，确保了与扩散模型实现的兼容性。
