# rocm/ 子模块索引

## 模块概述

`rocm/` 子模块实现了基于 AMD ROCm/GPU 的 OmniPlatform。ROCm 平台与 CUDA 平台结构类似，复用通用 GPU Worker，但在扩散注意力后端选择上使用 aiter 库代替 flash_attn。

## 文件列表

| 文件 | 说明 |
|------|------|
| [__init__.py.md](./__init__.py.md) | 包导出 |
| [platform.py.md](./platform.py.md) | RocmOmniPlatform 实现 |
| [stage_configs/index.md](./stage_configs/index.md) | ROCm 阶段配置 |

## 架构特点

- 与 CUDA 平台一样复用通用 GPU Worker（`GPUARWorker`、`GPUGenerationWorker`）。
- ROCm 上通过 `torch.cuda` 兼容层访问 AMD GPU（ROCm 的 PyTorch 端口将 AMD GPU 映射为 `cuda` 设备）。
- Flash Attention 支持依赖 aiter 库，仅在 gfx942/gfx950 架构上可用。
- 拥有自己的阶段配置目录 `vllm_omni/platforms/rocm/stage_configs`。
