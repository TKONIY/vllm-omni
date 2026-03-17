# `network_utils.py` — 网络工具函数

## 文件概述

`network_utils.py` 提供了网络相关的工具函数，当前仅包含端口可用性检测函数。

## 关键代码解析

### is_port_available — 端口可用性检测

```python
def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False
```

通过尝试绑定和监听来检测端口是否可用。设置 `SO_REUSEADDR` 避免 TIME_WAIT 状态的端口影响检测。捕获 `OverflowError` 处理端口号超出合法范围的情况。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `is_port_available` | 函数 | 检测指定端口是否可用 |

## 与其他模块的关系

- 被 `data.py` 中 `OmniDiffusionConfig.settle_port` 方法调用，在分布式初始化时寻找可用端口。

## 总结

`network_utils.py` 提供了端口可用性检测功能，是分布式环境初始化时自动寻找可用端口的基础工具。
