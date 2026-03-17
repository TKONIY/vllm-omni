# `utils.py` — 表格格式化与辅助函数

## 文件概述

该文件提供指标模块所需的工具函数，包括数据类字段定义生成、数据行构建、表格格式化输出，以及从引擎输出统计 token 数的辅助函数。

## 关键代码解析

### 字段定义生成

```python
def _build_field_defs(
    cls: type,
    exclude: set[str],
    transforms: dict[str, tuple[str, Callable]] | None = None,
) -> list[tuple[str, Callable[[Any], Any]]]:
```

从数据类自动生成字段定义列表。每个字段定义是 `(display_name, getter_fn)` 元组：
- 排除 `exclude` 中的字段
- 对 `transforms` 中指定的字段应用名称替换和值转换（如 bytes -> kbytes）
- 使用闭包捕获变量避免晚期绑定问题

### 数据行构建

```python
def _build_row(evt: Any, field_defs: list[tuple[str, Callable]]) -> dict[str, Any]:
    return {name: getter(evt) for name, getter in field_defs}
```

从事件对象和字段定义构建一行数据字典。

### 表格格式化

```python
def _format_table(
    title: str,
    data: dict[str, Any] | list[dict[str, Any]],
    value_fields: list[str],
    column_key: str | None = None,
    column_prefix: str = "",
) -> str:
```

基于 `PrettyTable` 的表格格式化，支持两种模式：

1. **单列模式**：`data` 为字典，显示为 `Field | Value` 两列
2. **多列模式**：`data` 为字典列表，每个字典一列，用于对比多个阶段或传输边

值格式化规则：
- 布尔值：直接转字符串
- 整数：千分位分隔（如 `1,234`）
- 浮点数：三位小数加千分位（如 `1,234.567`）
- 列表：逗号分隔的浮点数

### Token 计数

```python
def count_tokens_from_outputs(engine_outputs: list[Any]) -> int:
    total = 0
    for _ro in engine_outputs:
        outs = getattr(_ro, "outputs", None)
        if outs and len(outs) > 0:
            tokens = getattr(outs[0], "token_ids", None)
            if tokens is not None:
                total += len(tokens)
    return total
```

遍历引擎输出列表，从每个输出的 `outputs[0].token_ids` 中统计 token 总数。使用防御性编程（try/except + getattr）确保在任何输出格式异常时不会崩溃。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_build_field_defs(cls, exclude, transforms)` | 函数 | 从数据类自动生成字段定义 |
| `_build_row(evt, field_defs)` | 函数 | 构建数据行字典 |
| `_get_field_names(field_defs)` | 函数 | 提取字段名列表 |
| `_format_table(title, data, ...)` | 函数 | 格式化 PrettyTable 表格 |
| `count_tokens_from_outputs(engine_outputs)` | 函数 | 从引擎输出统计 token 数 |

## 与其他模块的关系

- **被 stats.py 使用**: `_build_field_defs`、`_build_row`、`_format_table` 均被 `OrchestratorAggregator.build_and_log_summary` 调用。
- **被 __init__.py 导出**: `count_tokens_from_outputs` 是模块公开 API 的一部分。
- **依赖 prettytable**: 使用 `PrettyTable` 库生成 ASCII 表格。

## 总结

`utils.py` 提供了指标模块的基础工具层，核心是一套配置驱动的表格格式化方案：通过字段定义自动提取数据类中的指标字段，应用可配置的转换和过滤，最终生成清晰的 ASCII 表格用于日志输出。`count_tokens_from_outputs` 则提供了引擎输出的 token 统计能力。
