# parallel/ — 并行注意力策略

## 模块概述

`parallel/` 提供与注意力计算内核正交的**通信/重分片策略**。目标是让 `Attention` 层保持简洁和可扩展：添加新的并行方式只需实现策略接口并在工厂中注册，无需修改核心 `Attention` 类。

## 支持的策略

| 策略 | 类 | 通信模式 | 说明 |
|------|-----|---------|------|
| 无并行 | `NoParallelAttention` | 无 | 单设备或 SP 未激活时的默认直通 |
| Ulysses | `UlyssesParallelAttention` | AllToAll | 序列分片 ↔ 头分片双向变换 |
| Ring | `RingParallelAttention` | P2P Ring | K/V 环形传递，本地 Q 不动 |
| Ulysses + Ring 混合 | Ulysses 管理 | AllToAll + P2P | Ulysses 做头分片，Ring 做 K/V 传递 |

## 策略选择流程

```
build_parallel_attention_strategy()
  ├── 无前向上下文 → NoParallelAttention
  ├── SP world_size <= 1 → NoParallelAttention
  ├── ulysses_degree > 1 → UlyssesParallelAttention（含混合模式）
  ├── ring_degree > 1 → RingParallelAttention
  └── 其他 → NoParallelAttention
```

## 三阶段流水线

```python
# 1. 预处理（通信 / 重分片）
query, key, value, metadata, ctx = strategy.pre_attention(query, key, value, metadata)

# 2. 内核执行（计算注意力）
output = attention_kernel(query, key, value, metadata)

# 3. 后处理（逆向通信）
output = strategy.post_attention(output, ctx)
```

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](./__init__.md) | 包初始化，导出核心接口 |
| [`base.py`](./base.md) | 策略接口 Protocol、上下文数据类、默认直通策略 |
| [`factory.py`](./factory.md) | 策略工厂，根据配置和环境选择策略 |
| [`ring.py`](./ring.md) | Ring 并行策略（输入准备 + 内核调度 + 后端回退） |
| [`ulysses.py`](./ulysses.md) | Ulysses 并行策略（AllToAll + 联合注意力处理） |

## 联合注意力（Joint Attention）处理

在文本+图像的多模态场景中，文本条件作为"联合张量"参与注意力计算：

| 策略 | Joint Query 处理 | Joint Key/Value 处理 |
|------|------------------|---------------------|
| Ulysses | 头切分后拼接到 query | 头切分后拼接到 key/value（纯 Ulysses）或保留给 Ring（混合模式） |
| Ring | 拼接到 query | 保留在元数据中，由 Ring 内核在 step=0 时处理 |
