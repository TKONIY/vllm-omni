# `ring_globals.py` — Ring Attention 全局依赖检测

## 文件概述

`ring_globals.py` 负责检测 Ring Attention 所需的各种注意力计算库是否可用。它在模块导入时执行依赖检测，并导出布尔标志和函数引用，供 `ring_kernels.py` 和 `ring_selector.py` 使用。

## 关键代码解析

### 1. Flash Attention 2 检测

```python
try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    HAS_FLASH_ATTN = True
except (ImportError, ModuleNotFoundError):
    HAS_FLASH_ATTN = False
```

FA2 的检测要求能导入底层 `_flash_attn_forward` 函数（而非高层 API），因为 Ring Attention 需要获取 `softmax_lse` 以正确累积分块结果。

### 2. Flash Attention 3 检测（双源回退）

```python
HAS_FA3 = False
fa3_fwd_func = None
fa3_attn_func = None

# 源 1: flash_attn_interface（源码编译）
try:
    from flash_attn_interface import _flash_attn_forward as fa3_fwd_func
    from flash_attn_interface import flash_attn_func as fa3_attn_func
    HAS_FA3 = True
except (ImportError, ModuleNotFoundError):
    pass

# 源 2: fa3_fwd_interface（PyPI 包）
if not HAS_FA3:
    try:
        from fa3_fwd_interface import _flash_attn_forward as fa3_fwd_func
        from fa3_fwd_interface import flash_attn_func as fa3_attn_func
        HAS_FA3 = True
    except (ImportError, ModuleNotFoundError):
        pass
```

FA3 支持两种安装方式：
1. 从 flash-attention 源码编译：通过 `flash_attn_interface` 导入
2. PyPI 包 `fa3-fwd`：通过 `fa3_fwd_interface` 导入（支持 Ampere/Ada/Hopper）

### 3. 其他后端检测

```python
HAS_FLASHINFER = False    # FlashInfer 检测
HAS_AITER = False         # AMD Aiter 检测
HAS_SAGE_ATTENTION = False    # SageAttention 检测
HAS_SPARSE_SAGE_ATTENTION = False  # SparseSageAttention 检测
HAS_NPU = False           # 昇腾 NPU 检测
```

每个库的检测模式一致：尝试导入 → 设置布尔标志 → 失败时设为 False。FlashInfer 特殊处理了运行时错误（版本不匹配等）。

### 4. 遗留别名

```python
HAS_FLASH_ATTN_HOPPER = HAS_FA3
flash_attn_forward_hopper = fa3_fwd_func
flash3_attn_func = fa3_attn_func
```

为向后兼容保留的别名。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `HAS_FLASH_ATTN` | bool | Flash Attention 2 是否可用 |
| `HAS_FA3` | bool | Flash Attention 3 是否可用 |
| `fa3_fwd_func` | 函数/None | FA3 低层 forward 函数 |
| `fa3_attn_func` | 函数/None | FA3 高层 attention 函数 |
| `HAS_FLASHINFER` | bool | FlashInfer 是否可用 |
| `HAS_AITER` | bool | AMD Aiter 是否可用 |
| `HAS_SAGE_ATTENTION` | bool | SageAttention 是否可用 |
| `HAS_SPARSE_SAGE_ATTENTION` | bool | SparseSageAttention 是否可用 |
| `HAS_NPU` | bool | 昇腾 NPU 是否可用 |

## 与其他模块的关系

- **`ring_kernels.py`**：导入可用性标志和函数，条件性地使用各后端
- **`ring_selector.py`**：导入可用性标志决定哪些 `AttnType` 可使用
- **`parallel/ring.py`**：检查 `HAS_FA3` 和 `HAS_FLASH_ATTN` 决定 Ring Attention 的内核

## 总结

`ring_globals.py` 是 Ring Attention 子系统的依赖检测中心。它在模块加载时探测所有可能的注意力计算库（FA2、FA3、FlashInfer、Aiter、SageAttention 等），导出布尔标志和函数引用，使得上层代码可以在运行时根据可用库选择最优的计算内核。
