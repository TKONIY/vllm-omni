# flux2_klein/ -- Flux 2 Klein 模型目录索引

## 目录概述

Flux 2 Klein 在 Flux 2 基础上增加了序列并行（Sequence Parallel）支持，适用于高分辨率图像生成。

## 文件列表

| 文件 | 说明 | 文档 |
|------|------|------|
| [`__init__.py`](__init__.md) | 包初始化 |
| [`flux2_klein_transformer.py`](flux2_klein_transformer.md) | SP 版 Transformer（_sp_plan 声明式配置） |
| [`pipeline_flux2_klein.py`](pipeline_flux2_klein.md) | Qwen3 编码器 + CFG 并行 Pipeline |

## 序列并行设计

```
_sp_plan:
  根层: hidden_states 沿 dim=1 分片 (auto_pad)
  rope_prepare: 图像 cos/sin 分片, 文本 cos/sin 复制
  proj_out: 沿 dim=1 聚合输出

SP Joint Attention:
  - 文本 KV 全量复制 (joint_strategy="front")
  - 图像 QKV 按 SP rank 分片
  - padding mask 处理不整除情况
```

## 相较 Flux 2 的区别

| 方面 | Flux 2 | Flux 2 Klein |
|------|--------|-------------|
| 序列并行 | 不支持 | Ulysses/Ring SP |
| 文本编码器 | Mistral3 | Qwen3 |
| RoPE 模块 | 内联计算 | `Flux2RopePrepare` 封装 |
| 注意力 | 标准 | SP 感知 Joint Attention |

## 总结

Klein 变体通过声明式 `_sp_plan` 和 SP 感知注意力，实现了对高分辨率图像生成的多 GPU 序列并行支持。图像 token 自动分片到多个 GPU，文本 token 复制到所有 GPU。
