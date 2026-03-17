# `hf_utils.py` — HuggingFace 模型检测工具

## 文件概述

`hf_utils.py` 提供了 HuggingFace 扩散模型的检测和配置加载工具。核心功能是判断给定的模型路径是否为扩散模型（通过检测 `model_index.json` 文件），支持本地路径和远程仓库两种场景。

## 关键代码解析

### is_diffusion_model — 扩散模型检测

```python
@lru_cache
def is_diffusion_model(model_name: str) -> bool:
    # 策略 1：检查本地 model_index.json（最快）
    if os.path.isdir(model_name):
        model_index_path = os.path.join(model_name, "model_index.json")
        if os.path.exists(model_index_path):
            config_dict = json.load(open(model_index_path))
            if config_dict.get("_class_name") and config_dict.get("_diffusers_version"):
                return True

    # 策略 2：通过 vLLM 工具获取 model_index.json（支持远程模型）
    config_dict = get_hf_file_to_dict("model_index.json", model_name)
    if config_dict is not None and config_dict.get("_class_name"):
        return True

    # 策略 3：尝试 diffusers 标准加载（最慢，可能有依赖问题）
    try:
        load_diffusers_config(model_name)
        return True
    except Exception:
        pass

    # 特殊检测：Bagel 模型（非 diffusers 格式）
    return _looks_like_bagel(model_name)
```

三层回退策略确保了最大兼容性：
1. 本地文件系统检查（最快，无需网络和 import）
2. vLLM 的 `get_hf_file_to_dict` 工具（支持本地和远程）
3. diffusers 标准加载（完整但可能因依赖冲突失败）

### _looks_like_bagel — Bagel 模型检测

```python
def _looks_like_bagel(model_name: str) -> bool:
    cfg = get_hf_file_to_dict("config.json", model_name)
    model_type = cfg.get("model_type")
    if model_type == "bagel":
        return True
    architectures = cfg.get("architectures") or []
    return "BagelForConditionalGeneration" in architectures
```

Bagel 是一种非 diffusers 格式的扩散模型，通过 `config.json` 中的 `model_type` 或 `architectures` 字段进行检测。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `is_diffusion_model` | 函数 | 判断模型是否为扩散模型，带 LRU 缓存 |
| `load_diffusers_config` | 函数 | 使用 diffusers 加载模型配置 |
| `_looks_like_bagel` | 函数 | 检测非 diffusers 格式的 Bagel 模型 |

## 与其他模块的关系

- 被 vLLM-Omni 的入口模块（如 `engine/`）调用，用于在启动时判断模型类型并选择正确的推理引擎
- 使用 vLLM 的 `get_hf_file_to_dict` 工具访问模型文件

## 总结

`hf_utils.py` 通过多层回退策略实现了可靠的扩散模型检测。`is_diffusion_model` 是模型路由的关键入口，决定模型走扩散推理路径还是自回归推理路径。`lru_cache` 缓存避免了重复检测的开销。
