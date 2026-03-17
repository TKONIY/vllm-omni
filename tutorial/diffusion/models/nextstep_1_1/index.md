# NextStep-1.1 模型教程索引

## 模块概述

`nextstep_1_1` 模块实现了 StepFun 的 NextStep-1.1 图像生成模型。该模型采用独特的自回归 Flow Matching 架构，使用 LLaMA 骨干逐 token 生成图像，每步通过轻量级 FlowMatchingHead 进行 SDE 采样。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 模块入口 |
| [`modeling_flux_vae.py`](modeling_flux_vae.py.md) | 自定义 VAE（Flux 风格） |
| [`modeling_nextstep.py`](modeling_nextstep.py.md) | 主模型（LLaMA + 图像投影 + FM Head） |
| [`modeling_nextstep_heads.py`](modeling_nextstep_heads.py.md) | Flow Matching 采样头 |
| [`modeling_nextstep_llama.py`](modeling_nextstep_llama.py.md) | TP 感知的 LLaMA 组件 |
| [`pipeline_nextstep_1_1.py`](pipeline_nextstep_1_1.py.md) | 完整生成管线 |

## 架构特点

- **自回归 Flow Matching**：LLM 逐步生成图像 token，非标准扩散
- **SDE 采样**：FlowMatchingHead 使用随机微分方程采样
- **3-branch CFG**：支持文本 CFG + 图像 CFG 的三分支引导
- **CFG 并行**：通过 KV 缓存分割和 broadcast 实现高效 CFG 并行
- **TP 感知**：LLaMA 层使用 vLLM 的融合并行线性层
- **KV 缓存**：使用 StaticCache 管理自回归推理
