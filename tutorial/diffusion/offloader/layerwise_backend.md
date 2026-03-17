# `layerwise_backend.py` — 层级（块级）卸载后端

## 文件概述

`layerwise_backend.py` 实现了层级（block-wise）CPU 卸载，这是显存优化的核心技术。它采用滑动窗口策略：在任意时刻只有少量 transformer block 驻留在 GPU 上，通过异步预取（prefetch）实现计算与内存传输的重叠。

## 关键代码解析

### 1. LayerwiseOffloadHook — 核心 hook 实现

```python
class LayerwiseOffloadHook(ModelHook):
    def __init__(self, next_block, device, stream=None, pin_memory=True):
        self.next_block = next_block          # 下一个要预取的 block
        self.device = device
        self.copy_stream = stream             # 异步传输使用的 CUDA stream
        self._prefetch_done = None            # 同步事件
        self._prev_hook = None                # 反向链接，用于缓存跳过场景
```

每个 hook 实例绑定到一个 transformer block，负责在该 block 执行前预取下一个 block 的权重。

### 2. 权重压缩与 CPU 存储

```python
@staticmethod
def _to_cpu(params, bufs, device, pin_memory=True):
    dtype_grouped_weights: dict[torch.dtype, dict[str, torch.Tensor]] = {}
    for name, param_or_buf in chain(params.items(), bufs.items()):
        dtype_grouped_weights.setdefault(param_or_buf.dtype, {})[name] = param_or_buf

    for dtype, name2weights in dtype_grouped_weights.items():
        total_numel = sum(t.numel() for t in name2weights.values())
        cpu_tensor = torch.empty(total_numel, dtype=dtype, device="cpu", pin_memory=pin_memory)
        # 将所有参数展平复制到连续 CPU 张量中
        for name, param_or_buf in name2weights.items():
            cpu_tensor[offset:offset+numel].copy_(param_or_buf.flatten())
            param_or_buf.data = torch.empty((), device=device, dtype=dtype)  # 释放 GPU 内存
```

将一个 block 的所有参数按 dtype 分组后展平到连续的 CPU 张量中。使用页锁定内存（pin_memory）加速后续的 CPU->GPU 传输。原始参数被替换为空张量以释放 GPU 显存。

### 3. 异步预取机制

```python
@torch.compiler.disable
def prefetch_layer(self, non_blocking=True):
    self.copy_stream.wait_stream(current_omni_platform.current_stream())

    with current_omni_platform.stream(self.copy_stream):
        for dtype, cpu_weight in self.dtype_cpu_flattened_weights.items():
            gpu_weight = torch.empty(cpu_weight.shape, dtype=dtype, device=self.device)
            gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)

        evt.record(self.copy_stream)

    # 将 GPU 数据切片赋值回各参数
    for dtype, ordered_metadata in self.dtype_metadata.items():
        for metadata in ordered_metadata:
            target_param_or_buf.data = gpu_weight[offset:offset+numel].view(shape)

    self._prefetch_done = evt
```

在独立的 CUDA stream 上异步执行 CPU->GPU 拷贝，不阻塞主计算流。通过 CUDA event 实现同步。

### 4. Hook 的 pre/post_forward

```python
def pre_forward(self, module, *args, **kwargs):
    # 如果权重未就绪（被缓存系统跳过），同步预取
    if not self.is_materialized and self._prev_hook is not None:
        self._prev_hook.prefetch_layer(non_blocking=False)
    # 异步预取下一个 block
    self.prefetch_layer(non_blocking=True)
    return args, kwargs

def post_forward(self, module, output):
    # 释放当前 block 的 GPU 内存
    self.offload_layer()
    return output
```

执行流程：当前 block 计算开始前，启动下一个 block 的异步预取；当前 block 计算完成后，释放其 GPU 内存。

### 5. LayerWiseOffloadBackend — 后端管理

```python
class LayerWiseOffloadBackend(OffloadBackend):
    def enable(self, pipeline):
        modules = ModuleDiscovery.discover(pipeline)
        # 编码器和 VAE 常驻 GPU
        for enc in modules.encoders:
            enc.to(self.device)

        for dit_module in modules.dits:
            blocks = self.get_blocks_from_dit(dit_module)
            # 为每个 block 注册 hook，形成循环链
            last_hook = apply_block_hook(blocks[-1], blocks[0], ...)
            last_hook.prefetch_layer(non_blocking=False)  # 预加载第一个 block
            for i, block in enumerate(blocks[:-1]):
                hook = apply_block_hook(block, blocks[(i+1) % num_blocks], ...)
            # 建立反向链接用于缓存跳过场景
            for i in range(len(block_hooks)):
                block_hooks[i]._prev_hook = block_hooks[i-1]
```

后端初始化时：
1. 编码器和 VAE 常驻 GPU。
2. DiT 的非 block 模块（如 embedding、norm）常驻 GPU。
3. 每个 block 注册 hook，形成"最后一个 block 预取第一个 block"的循环链。
4. `_prev_hook` 反向链接用于处理 Cache-DiT 等缓存系统跳过 block 的场景。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `LayerwiseOffloadHook` | 类 | 块级卸载 hook，实现异步预取和内存释放 |
| `_to_cpu` | 静态方法 | 将参数展平到连续 CPU 张量并释放 GPU 内存 |
| `prefetch_layer` | 方法 | 异步预取 block 权重到 GPU |
| `offload_layer` | 方法 | 释放 block 的 GPU 内存 |
| `apply_block_hook` | 函数 | 为 block 注册卸载 hook |
| `remove_block_hook` | 函数 | 移除 block 的卸载 hook |
| `LayerWiseOffloadBackend` | 类 | 层级卸载后端管理器 |
| `get_blocks_from_dit` | 静态方法 | 从 DiT 模型获取 block 列表 |

## 与其他模块的关系

- **`base.py`**：`LayerWiseOffloadBackend` 继承 `OffloadBackend`。
- **`module_collector.py`**：使用 `ModuleDiscovery` 发现 pipeline 中的模块。
- **`hooks.py`**：使用 `HookRegistry` 和 `ModelHook` 基础设施。
- **DiT 模型**：通过 `_layerwise_offload_blocks_attr` 类属性声明 blocks 属性名。
- **Cache-DiT**：`_prev_hook` 机制兼容缓存系统跳过 block 的场景。

## 总结

层级卸载是该项目显存优化的核心技术。通过将 transformer block 的权重按需在 CPU 和 GPU 间滑动，使得大型扩散模型可以在有限显存下运行。关键优化包括：按 dtype 展平的连续 CPU 存储、页锁定内存、异步 stream 预取以及计算-传输重叠。反向链接机制确保了与 Cache-DiT 等高级缓存系统的兼容性。
