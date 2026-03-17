# `config/yaml_util.py` — YAML 工具封装

## 文件概述

`yaml_util.py` 封装了所有 OmegaConf 操作，作为项目中 YAML 配置处理的统一入口。其他模块应通过此文件的函数操作配置，而不是直接使用 OmegaConf。

## 关键代码解析

### load_yaml_config — 加载 YAML 文件

```python
def load_yaml_config(path: str | Any) -> DictConfig:
    return OmegaConf.load(path)
```

加载 YAML 文件并返回 `DictConfig`，支持属性风格访问（如 `config.stages`）。

### create_config — 创建配置对象

```python
def create_config(data: Any) -> DictConfig:
    return OmegaConf.create(data)
```

将 Python 字典或列表包装为 OmegaConf 配置对象。

### merge_configs — 深度合并配置

```python
def merge_configs(*cfgs: Any) -> dict:
    merged = OmegaConf.merge(*cfgs)
    return OmegaConf.to_container(merged, resolve=True)
```

从左到右深度合并多个配置，返回纯 Python 字典。支持 OmegaConf 插值解析。

### to_dict — 配置转字典

```python
def to_dict(obj: Any, *, resolve: bool = True) -> Any:
    return OmegaConf.to_container(obj, resolve=resolve)
```

将 OmegaConf 容器转换为纯 Python 数据结构，可选择是否解析插值。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `load_yaml_config` | 函数 | 加载 YAML 文件为 DictConfig |
| `create_config` | 函数 | 将字典包装为 DictConfig |
| `merge_configs` | 函数 | 深度合并多个配置 |
| `to_dict` | 函数 | 将 DictConfig 转为纯字典 |

## 与其他模块的关系

- 被 `stage_config.py` 用于加载和处理流水线 YAML
- 被 `config/__init__.py` 导出为公共 API
- 所有 OmegaConf 操作都应通过此模块进行

## 总结

`yaml_util.py` 通过统一封装 OmegaConf 操作，降低了项目对特定配置库的耦合度。如果未来需要替换 OmegaConf，只需修改此文件即可。
