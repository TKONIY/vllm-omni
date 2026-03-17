# transformers_utils 模块索引

本模块提供 vllm-omni 的 HuggingFace Transformers 配置扩展，为自定义模型（MammothModa2、Fish Speech）注册 `AutoConfig` 配置类。

## 模块结构

```
transformers_utils/
├── __init__.py                    # 空初始化文件
└── configs/
    ├── __init__.py                # 延迟加载与自动注册
    ├── fish_speech.py             # Fish Speech 配置注册
    └── mammoth_moda2.py           # MammothModa2 配置类定义与注册
```

## 文档列表

| 文件 | 说明 |
|------|------|
| [__init__.md](__init__.md) | 包初始化 |
| [configs/__init__.md](configs/__init__.md) | configs 包：延迟加载与注册 |
| [configs/fish_speech.md](configs/fish_speech.md) | Fish Speech 配置注册 |
| [configs/mammoth_moda2.md](configs/mammoth_moda2.md) | MammothModa2 配置类 |

## 模块间关系

- `configs/__init__.py` 通过延迟导入和即时导入两种方式确保配置类注册到 `AutoConfig`。
- `configs/mammoth_moda2.py` 定义了 MammothModa2 的完整配置层次，与 `tokenizers/mammoth_moda2_tokenizer.py` 配合使用。
- `configs/fish_speech.py` 注册 Fish Speech 的配置类，实际定义在 `model_executor` 模块中。
