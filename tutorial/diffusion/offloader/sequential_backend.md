# `sequential_backend.py` — 模型级顺序卸载后端

## 文件概述

`sequential_backend.py` 实现了模型级（组件级）的顺序 CPU 卸载。其核心思想是 DiT 和编码器互斥占用 GPU：当 DiT 执行前向传播时，编码器被卸载到 CPU；当编码器执行时，DiT 被卸载到 CPU。这比层级卸载粒度更粗，但实现更简单且开销更低。

## 关键代码解析

### 1. SequentialOffloadHook — 顺序卸载 hook

```python
class SequentialOffloadHook(ModelHook):
    def __init__(self, offload_targets, device, pin_memory=True):
        self.offload_targets = offload_targets  # 要卸载到 CPU 的模块列表
        self.device = device
        self.pin_memory = pin_memory

    def pre_forward(self, module, *args, **kwargs):
        # 将目标模块卸载到 CPU
        for target in self.offload_targets:
            self._to_cpu(target)
        # 将当前模块加载到 GPU
        self._to_gpu(module)
        current_omni_platform.synchronize()
        return args, kwargs
```

在每个模块的前向传播之前，先将互斥的目标模块移到 CPU，再将当前模块移到 GPU。这保证了同一时刻只有一个大型组件占用 GPU 显存。

### 2. 安全的参数移动

```python
@staticmethod
def _move_params(module, device):
    """逐参数移动，避免递归调用 module.to()"""
    for p in module.parameters():
        if p.data.device != device:
            p.data = p.data.to(device, non_blocking=True)
    for b in module.buffers():
        if b.device != device:
            b.data = b.data.to(device, non_blocking=True)
```

使用逐参数移动而非 `module.to(device)`，避免了递归问题。这是针对 Cache-DiT 等系统的 workaround——它们的 `CachedBlocks` 持有对原始 transformer 的引用，直接调用 `to()` 会导致循环引用和递归调用。

### 3. CPU 内存固定

```python
def _to_cpu(self, module):
    self._move_params(module, torch.device("cpu"))
    current_omni_platform.empty_cache()
    if self.pin_memory:
        for p in module.parameters():
            if p.data.device.type == "cpu" and not p.data.is_pinned():
                p.data = p.data.pin_memory()
```

卸载到 CPU 后立即清理 GPU 缓存，并将 CPU 上的参数固定（pin），加速后续重新加载到 GPU 时的传输速度。

### 4. 双向 hook 注册

```python
def apply_sequential_offload(dit_modules, encoder_modules, device, pin_memory=True):
    # DiT 上注册 hook：执行前卸载编码器
    for dit_mod in dit_modules:
        hook = SequentialOffloadHook(offload_targets=encoder_modules, device=device)
        registry.register_hook("sequential_offload", hook)

    # 编码器上注册 hook：执行前卸载 DiT
    for enc in encoder_modules:
        hook = SequentialOffloadHook(offload_targets=dit_modules, device=device)
        registry.register_hook("sequential_offload", hook)
```

双向注册确保无论哪个组件先执行，都能正确地腾出 GPU 空间。

### 5. ModelLevelOffloadBackend — 后端管理

```python
class ModelLevelOffloadBackend(OffloadBackend):
    def enable(self, pipeline):
        modules = ModuleDiscovery.discover(pipeline)
        # 编码器先移到 GPU
        for enc in modules.encoders:
            enc.to(self.device)
        # VAE 移到 GPU
        if modules.vae is not None:
            modules.vae.to(self.device, non_blocking=True)
        # 应用顺序卸载
        apply_sequential_offload(
            dit_modules=modules.dits,
            encoder_modules=modules.encoders,
            device=self.device,
        )
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `SequentialOffloadHook` | 类 | 顺序卸载 hook，实现互斥 GPU 占用 |
| `_move_params` | 静态方法 | 安全的逐参数设备移动 |
| `_to_cpu` | 方法 | 将模块卸载到 CPU 并固定内存 |
| `_to_gpu` | 方法 | 将模块加载到 GPU |
| `apply_sequential_offload` | 函数 | 在 DiT 和编码器之间建立双向卸载 hook |
| `remove_sequential_offload` | 函数 | 移除所有顺序卸载 hook |
| `ModelLevelOffloadBackend` | 类 | 模型级卸载后端管理器 |

## 与其他模块的关系

- **`base.py`**：`ModelLevelOffloadBackend` 继承 `OffloadBackend`。
- **`module_collector.py`**：使用 `ModuleDiscovery` 发现 DiT 和编码器。
- **`hooks.py`**：使用 `HookRegistry` 注册和管理 hook。
- **Cache-DiT**：`_move_params` 的设计考虑了 Cache-DiT 的循环引用问题。

## 总结

模型级顺序卸载适用于 GPU 显存不足以同时容纳 DiT 和编码器，但可以容纳其中任一的场景。相比层级卸载，它的切换频率更低（每次扩散步骤切换一次而非每个 block 切换），因此传输开销更小。但显存节省幅度也相应较小，因为每次需要将整个 DiT 或编码器加载到 GPU。
