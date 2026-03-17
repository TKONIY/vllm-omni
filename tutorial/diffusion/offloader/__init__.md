# `__init__.py` — 离线加载模块入口与后端工厂

## 文件概述

`offloader/__init__.py` 是 CPU 卸载（offloading）子模块的入口文件。它提供了 `get_offload_backend` 工厂函数，根据配置自动选择并创建合适的卸载后端。

## 关键代码解析

```python
def get_offload_backend(
    od_config: OmniDiffusionConfig,
    device: torch.device | None = None,
) -> OffloadBackend | None:
    # 从配置中提取并验证卸载设置
    config = OffloadConfig.from_od_config(od_config)

    # 无卸载请求
    if config.strategy == OffloadStrategy.NONE:
        return None

    # 验证平台支持（目前需要 CUDA）
    if not current_omni_platform.supports_cpu_offload() or current_omni_platform.get_device_count() < 1:
        return None

    # 自动检测设备
    if device is None:
        device = current_omni_platform.get_torch_device()

    # 根据策略创建后端
    if config.strategy == OffloadStrategy.MODEL_LEVEL:
        return ModelLevelOffloadBackend(config, device)
    elif config.strategy == OffloadStrategy.LAYER_WISE:
        return LayerWiseOffloadBackend(config, device)
```

工厂函数的流程：配置验证 -> 平台检查 -> 设备检测 -> 后端创建。返回 `None` 表示不启用卸载。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_offload_backend` | 函数 | 卸载后端工厂函数 |
| `OffloadBackend` | 类（重导出） | 卸载后端基类 |
| `OffloadConfig` | 类（重导出） | 卸载配置数据类 |
| `OffloadStrategy` | 枚举（重导出） | 卸载策略枚举 |
| `LayerWiseOffloadBackend` | 类（重导出） | 层级卸载后端 |
| `ModelLevelOffloadBackend` | 类（重导出） | 模型级卸载后端 |

## 与其他模块的关系

- **`base.py`**：导入 `OffloadBackend`、`OffloadConfig`、`OffloadStrategy`。
- **`layerwise_backend.py`**：导入 `LayerWiseOffloadBackend`。
- **`sequential_backend.py`**：导入 `ModelLevelOffloadBackend` 及辅助函数。
- **扩散 pipeline**：pipeline 初始化时调用 `get_offload_backend` 获取并启用卸载。

## 总结

此入口文件通过工厂模式屏蔽了卸载后端的选择细节。使用者只需传入配置对象，即可获得正确的卸载后端实例（或 `None`）。
