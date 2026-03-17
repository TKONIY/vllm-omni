# kv_transfer/ — KV 缓存传输补丁模块

## 模块概述

`kv_transfer/` 子模块通过 monkey-patch 技术修补 vLLM 原生的 `MooncakeConnector`，解决 Prefill-Decode (PD) 分离场景下 request ID 不匹配的问题。

## 问题背景

vLLM 的 `InputProcessor.assign_request_id()` 会给每个请求 ID 追加一个随机 8 字符后缀。在 PD 分离部署中，Prefill 引擎存储 KV 缓存时使用的是它自己的后缀，而 Decode 引擎生成的是另一个不同的后缀——导致 Decode 端永远找不到 Prefill 端写入的 KV 数据。

## 解决方案

补丁通过 `kv_transfer_params` 字典将 Prefill 引擎内部的 `remote_request_id` 传递给 Decode 端，使其能用正确的 key 检索 KV 数据。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 模块 docstring，描述补丁目的 |
| [`monkey_patch.py`](monkey_patch.py.md) | 补丁实现：`PatchedMooncakeConnector` 和 `apply_mooncake_connector_patch()` |
