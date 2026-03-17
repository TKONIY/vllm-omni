# `logo.py` — 启动 Logo 显示

## 文件概述

定义了 vLLM-Omni 的终端 ASCII 艺术 Logo，在服务启动时显示。Logo 使用 ANSI 颜色码实现彩色输出，当终端不支持颜色时自动降级为纯文本。

## 关键代码解析

```python
# 颜色定义
ORANGE = "\033[38;5;208m"
BLUE = "\033[34m"
WHITE = "\033[97m"
PURPLE = "\033[35m"
RESET = "\033[0m"

# Logo 由 vLLM + "O" + "MNI" 三部分组成
LOGO = f"""{VLLM_L1}{GAP_L1}{O_L1}{MNI_L1}
{VLLM_L2}{GAP_L2}{O_L2}{MNI_L2}
...
"""

def log_logo() -> None:
    logo = LOGO if current_formatter_type(logger) == "color" else _ANSI_RE.sub("", LOGO)
    logger.info(logo)
```

`log_logo()` 检查当前日志格式化器是否支持颜色，如果不支持则用正则表达式去除所有 ANSI 转义序列。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `LOGO` | 常量 | 带颜色的 ASCII 艺术 Logo |
| `log_logo()` | 函数 | 智能显示 Logo（自动适配终端能力） |

## 与其他模块的关系

- 被 `serve.py` 中的 `OmniServeCommand.cmd()` 在服务启动时调用

## 总结

一个纯展示性质的模块，为 vLLM-Omni 服务启动提供品牌标识，具备终端颜色能力自动检测功能。
