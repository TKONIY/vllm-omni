# `utils.py` — 稳定 LoRA ID 生成工具

## 文件概述

该文件提供 `stable_lora_int_id` 函数，用于为 LoRA 适配器生成确定性的正整数 ID。解决了 Python 内置 `hash()` 跨进程不稳定的问题。

## 关键代码解析

```python
def stable_lora_int_id(lora_path: str) -> int:
    """Return a deterministic positive integer ID for a LoRA adapter."""
    digest = hashlib.sha256(lora_path.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)
    return value or 1
```

实现细节：
1. 对 LoRA 路径字符串计算 SHA-256 哈希
2. 取前 8 字节（64 位）转为无符号整数
3. 通过 `& ((1 << 63) - 1)` 确保为 63 位正整数
4. 如果结果为 0，返回 1（避免无效 ID）

**为什么不用 `hash()`？**
Python 的 `hash()` 函数默认启用了 `PYTHONHASHSEED` 随机化，每次进程启动结果不同。而 vLLM 使用 `lora_int_id` 作为适配器的缓存键，需要跨进程保持一致，因此需要基于 SHA-256 的确定性方案。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `stable_lora_int_id(lora_path)` | 函数 | 从路径生成稳定的 63 位正整数 ID |

## 与其他模块的关系

- **被 LoRA 加载流程使用**: 在加载 LoRA 适配器时调用，生成缓存键。
- **无外部依赖**: 仅使用标准库 `hashlib`。

## 总结

该文件解决了一个具体但关键的问题：在多进程环境下为 LoRA 适配器生成稳定的整数标识符。通过 SHA-256 哈希方案，确保相同路径在任何进程、任何时刻都映射到相同的 ID。
