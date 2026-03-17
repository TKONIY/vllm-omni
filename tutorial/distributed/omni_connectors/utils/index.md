# utils/ — 工具模块

## 模块概述

`utils/` 子模块提供 OmniConnector 框架的基础工具：配置解析、序列化/反序列化、日志、初始化流程和 KV 缓存处理工具。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 空初始化文件 |
| [`config.py`](config.py.md) | 配置数据类：`ConnectorSpec`、`OmniTransferConfig` |
| [`serialization.py`](serialization.py.md) | 序列化框架：`OmniMsgpackEncoder/Decoder`、`OmniSerializer` |
| [`logging.py`](logging.py.md) | 日志工具：`get_connector_logger` |
| [`initialization.py`](initialization.py.md) | 初始化工具：配置加载、连接器创建、阶段配置解析 |
| [`kv_utils.py`](kv_utils.py.md) | KV 缓存工具：`normalize_layer_kv` |
