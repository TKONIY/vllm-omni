# `utils.py` — 通用工具函数

## 文件概述

该文件提供了阶段配置加载、解析和验证的高层工具函数，是多阶段管线初始化流程的核心。同时包含数据类过滤、容器检测和 PID 命名空间检测等辅助功能。

## 关键代码解析

### 模型配置路径解析

```python
def resolve_model_config_path(model: str) -> str:
    """根据模型类型解析阶段配置文件路径"""
    # 1. 尝试标准 transformers 格式
    hf_config = get_config(model, trust_remote_code=True)
    model_type = hf_config.model_type

    # 2. 尝试设备特定配置目录
    complete_config_path = PROJECT_ROOT / default_config_path / f"{model_type}.yaml"
    if os.path.exists(complete_config_path):
        return str(complete_config_path)

    # 3. 回退到默认配置目录
    stage_config_path = PROJECT_ROOT / f"vllm_omni/model_executor/stage_configs/{model_type}.yaml"
```

通过模型的 `model_type` 自动查找对应的阶段配置 YAML 文件，支持设备特定配置（如 NPU 专用配置）。

### 阶段配置加载

```python
def load_stage_configs_from_yaml(config_path: str, base_engine_args=None) -> list:
    """从 YAML 文件加载阶段配置"""
    config_data = load_yaml_config(config_path)
    stage_args = config_data.stage_args
    base_engine_args = _convert_dataclasses_to_dict(base_engine_args)
    base_engine_args = create_config(base_engine_args)
    for stage_arg in stage_args:
        # 合并全局引擎参数和阶段特定参数
        base_engine_args_tmp = merge_configs(base_engine_args_tmp, stage_arg.engine_args)
        # 注入运行时配置
        if hasattr(stage_arg, "runtime"):
            base_engine_args_tmp["max_num_seqs"] = int(runtime_cfg.get("max_batch_size", 1))
    return stage_args
```

YAML 配置加载时会将全局引擎参数与每个阶段的特定参数合并，确保每个阶段都继承了公共配置。

### 最终阶段 ID 计算

```python
def get_final_stage_id_for_e2e(output_modalities, default_modalities, stage_list):
    """确定端到端指标应使用哪个阶段作为终点"""
    for _sid in range(last_stage_id, -1, -1):
        if (getattr(stage_list[_sid], "final_output", False)
            and stage_list[_sid].final_output_type in output_modalities):
            return _sid
```

从后向前扫描阶段列表，找到最后一个匹配请求输出模态的 `final_output` 阶段。

### OmegaConf 兼容性

```python
def _convert_dataclasses_to_dict(obj):
    """递归将 dataclass、Counter、set 等转为 OmegaConf 兼容类型"""
    if obj.__class__.__name__ == "Counter":
        return dict(obj)
    if isinstance(obj, set):
        return list(obj)
    if is_dataclass(obj):
        return _convert_dataclasses_to_dict(asdict(obj))
    if callable(obj):
        return None  # 过滤掉不可序列化的可调用对象
```

处理 Python 原生类型与 OmegaConf 配置系统之间的兼容性问题。

### 容器和 PID 检测

```python
def in_container() -> bool:
    """检测是否在容器中运行"""
    if os.path.exists("/.dockerenv"):
        return True
    cg = _read_text("/proc/1/cgroup") or ""
    return any(m in cg for m in ("docker", "containerd", "kubepods"))

def has_pid_host() -> bool | None:
    """检测是否使用 --pid=host"""
    comm2 = _read_text("/proc/2/comm")
    if comm2 and comm2.strip() == "kthreadd":
        return True
```

用于决定是否可以使用基于进程 ID 的内存跟踪，或需要使用顺序初始化锁。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `resolve_model_config_path()` | 函数 | 根据模型类型查找配置文件路径 |
| `load_stage_configs_from_model()` | 函数 | 从模型加载阶段配置 |
| `load_stage_configs_from_yaml()` | 函数 | 从 YAML 加载阶段配置 |
| `load_and_resolve_stage_configs()` | 函数 | 带回退的统一配置加载入口 |
| `get_final_stage_id_for_e2e()` | 函数 | 计算 E2E 指标的最终阶段 |
| `inject_omni_kv_config()` | 函数 | 注入连接器配置到阶段引擎参数 |
| `filter_dataclass_kwargs()` | 函数 | 按 dataclass 字段过滤 kwargs |
| `build_base_engine_args()` | 函数 | 构建基础引擎参数（tokenizer + 并行配置） |
| `in_container()` / `has_pid_host()` | 函数 | 容器环境检测 |

## 与其他模块的关系

- 被 `OmniBase` 和 `AsyncOmniEngine` 在初始化时调用
- 依赖 `vllm_omni.config.yaml_util` 进行 YAML 配置解析
- `inject_omni_kv_config()` 被连接器初始化流程使用
- `get_final_stage_id_for_e2e()` 被 `OmniBase._compute_final_stage_id()` 调用

## 总结

`utils.py` 是多阶段管线初始化的关键工具集，负责从模型类型自动解析配置、加载和合并 YAML 阶段配置、处理类型兼容性问题。它还提供了容器环境检测功能，用于在不同部署环境中选择合适的资源管理策略。
