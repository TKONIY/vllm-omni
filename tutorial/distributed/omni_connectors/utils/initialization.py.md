# `initialization.py` — 初始化与配置加载工具

## 文件概述

该文件提供了 OmniConnector 系统的配置加载和初始化流程，包括从 JSON/YAML 配置文件解析连接器配置、为特定阶段构建连接器、以及 Orchestrator 和 Stage 级别的初始化函数。

## 关键代码解析

### 1. load_omni_transfer_config — 配置加载核心

```python
def load_omni_transfer_config(config_path=None, config_dict=None, default_shm_threshold=65536):
```

支持 JSON 和 YAML 格式。配置解析流程：

1. **全局连接器**（`runtime.connectors`）：定义共享的连接器规格
2. **阶段级连接器**（`stage_args[].input_connectors` / `output_connectors`）：
   - 可以引用全局连接器（字符串引用）或内联定义
   - 同一边的两端定义必须类型一致，否则报错
3. **自动配置**：缺失边自动创建 `SharedMemoryConnector`
   - 从 `runtime.edges` 推断
   - 从 `engine_input_source` 推断
4. **一致性检查**：如果任何期望的边没有连接器，抛出 `ValueError`

```python
# 自动配置示例
if edge_key not in connectors:
    connectors[edge_key] = ConnectorSpec(
        name="SharedMemoryConnector",
        extra={"shm_threshold_bytes": default_shm_threshold},
    )
```

### 2. get_connectors_config_for_stage — 阶段配置提取

```python
def get_connectors_config_for_stage(transfer_config, stage_id) -> dict:
```

从全局配置中提取特定阶段相关的连接器配置，注入角色信息：
- 入边（`to_stage == target_stage`）：注入 `role="receiver"`
- 出边（`from_stage == target_stage`，仅 stage 0）：注入 `role="sender"`

### 3. build_stage_connectors — 阶段连接器构建

```python
def build_stage_connectors(stage_id, connectors_config) -> dict:
```

仅实例化 INPUT 连接器（`from_stage_X`），因为 Stage Worker 只通过连接器接收数据。OUTPUT 连接器由 Orchestrator 层处理。

### 4. initialize_orchestrator_connectors — Orchestrator 初始化

```python
def initialize_orchestrator_connectors(config_path, worker_backend=None, shm_threshold_bytes=65536):
```

Ray 后端下将 `shm_threshold_bytes` 设为 `sys.maxsize`，有效禁用共享内存（使用 Ray 内置传输）。

### 5. resolve_omni_kv_config_for_stage — KV 配置解析

```python
def resolve_omni_kv_config_for_stage(transfer_cfg, stage_id):
```

解析特定阶段的连接器配置用于 KV 缓存传输：
- 优先出边（Sender 角色）
- 回退到入边（Receiver 角色）
- 注入 `role` 到连接器配置中

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `load_omni_transfer_config()` | function | 从文件/字典加载全局传输配置 |
| `initialize_connectors_from_config()` | function | 加载配置并创建所有连接器实例 |
| `create_connectors_from_config()` | function | 从连接器规格字典批量创建连接器 |
| `get_connectors_config_for_stage()` | function | 提取阶段级连接器配置 |
| `build_stage_connectors()` | function | 为阶段构建输入连接器 |
| `initialize_orchestrator_connectors()` | function | Orchestrator 级初始化 |
| `get_stage_connector_config()` | function | 获取阶段连接器配置（带异常处理） |
| `resolve_omni_kv_config_for_stage()` | function | 解析阶段 KV 缓存传输配置 |

## 与其他模块的关系

- 使用 `OmniConnectorFactory` 创建连接器
- 使用 `ConnectorSpec` / `OmniTransferConfig` 配置类
- 被 Orchestrator 和 Stage Worker 的启动流程调用
- 被 `distributed/__init__.py` 导出

## 总结

`initialization.py` 是连接器系统的"引导程序"，负责将配置文件转换为运行时的连接器实例。它支持灵活的配置方式（全局/阶段级/自动推断），处理了 Ray 后端的特殊需求，并通过一致性检查确保配置完整性。
