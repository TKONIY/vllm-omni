# ring/ — Ring Attention 底层组件

## 模块概述

`ring/` 包含 Ring Attention 算法所需的底层组件，包括全局依赖检测、计算内核集合、内核选择器和分块结果合并工具。这些组件被上层的 `ring_flash_attn.py` 和 `ring_pytorch_attn.py` 使用。

## 模块关系

```
ring_flash_attn.py / ring_pytorch_attn.py
  ↓ 使用
ring_selector.py  ──→  ring_kernels.py  ──→  ring_globals.py
                            ↑
ring_utils.py ←─── 被 ring_flash_attn.py 调用
```

- **ring_globals.py**：检测所有可用的注意力库（FA2/FA3/FlashInfer/Aiter/Sage 等），导出布尔标志
- **ring_kernels.py**：实现各库的统一调用接口，每个内核返回 `(output, lse)` 元组
- **ring_selector.py**：定义 `AttnType` 枚举和选择函数，根据类型返回对应内核
- **ring_utils.py**：实现在线 softmax 合并公式，正确累积分块注意力结果

## Ring Attention 算法流程

```
对于 world_size 个步骤：
  1. 异步发送当前 K/V → 下一设备，接收上一设备的 K/V
  2. 使用 select_flash_attn_impl 选择的内核计算 attn(Q, K_step, V_step) → (block_out, block_lse)
  3. 使用 update_out_and_lse 合并到全局结果
  4. 等待通信完成，更新 K/V
```

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](./__init__.md) | 包初始化 |
| [`ring_globals.py`](./ring_globals.md) | 全局依赖检测（FA2/FA3/FlashInfer/Aiter/Sage/NPU） |
| [`ring_kernels.py`](./ring_kernels.md) | 5 种注意力计算内核的统一接口 |
| [`ring_selector.py`](./ring_selector.md) | 11 种 AttnType 枚举与内核选择器 |
| [`ring_utils.py`](./ring_utils.md) | 在线 softmax 分块合并工具（核心数学） |

## 核心数学

Ring Attention 的正确性依赖于在线 softmax 合并：

```
new_out = out - sigmoid(block_lse - lse) * (out - block_out)
new_lse = lse - logsigmoid(lse - block_lse)
```

此公式等价于对全局 softmax 的增量计算，保证了数值稳定性和数学等价性。
