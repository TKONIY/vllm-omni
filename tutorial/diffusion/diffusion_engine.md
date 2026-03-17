# `diffusion_engine.py` — 扩散推理引擎

## 文件概述

`diffusion_engine.py` 是扩散模型推理的顶层入口，定义了 `DiffusionEngine` 类。它整合了模型注册、执行器管理、前后处理、性能分析等功能，为上层提供统一的推理接口。用户通过 `DiffusionEngine.step()` 提交扩散请求并获取结果。

## 关键代码解析

### DiffusionEngine 初始化

```python
class DiffusionEngine:
    def __init__(self, od_config: OmniDiffusionConfig):
        self.od_config = od_config
        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)
        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)
        try:
            self._dummy_run()
        except Exception as e:
            self.close()
            raise e
```

初始化流程：
1. 从 `registry.py` 获取模型特定的前后处理函数
2. 根据配置选择并创建执行器（如多进程执行器）
3. 执行 `_dummy_run()` 预热模型

### step() — 核心推理方法

```python
def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
    # 1. 前处理
    if self.pre_process_func is not None:
        request = self.pre_process_func(request)
    # 2. 提交并等待执行结果
    output = self.add_req_and_wait_for_response(request)
    # 3. 后处理（如 VAE 解码、格式转换等）
    outputs = self.post_process_func(output.output)
    # 4. 封装为 OmniRequestOutput 列表
    return [OmniRequestOutput.from_diffusion(...)]
```

该方法处理了单请求和批量请求两种场景，支持图像和音频输出类型，并记录了详细的性能计时信息。

### 辅助功能函数

```python
def supports_image_input(model_class_name: str) -> bool:
def supports_audio_input(model_class_name: str) -> bool:
def supports_audio_output(model_class_name: str) -> bool:
def image_color_format(model_class_name: str) -> str:
```

通过查询模型注册表来检测模型能力，用于在 dummy run 和输出处理中做条件判断。

### 性能分析接口

```python
def start_profile(self, trace_filename: str | None = None) -> None:
def stop_profile(self) -> dict:
```

通过 `collective_rpc` 向所有 worker 广播 profiler 控制指令，支持 torch profiler 的分布式收集。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionEngine` | 类 | 扩散推理引擎，统一管理执行器、前后处理和性能分析 |
| `DiffusionEngine.step` | 方法 | 核心推理接口，提交请求并返回结果 |
| `DiffusionEngine.make_engine` | 静态方法 | 工厂方法，创建引擎实例 |
| `DiffusionEngine.collective_rpc` | 方法 | 向所有 worker 发起 RPC 调用 |
| `supports_image_input` | 函数 | 检查模型是否支持图像输入 |
| `supports_audio_output` | 函数 | 检查模型是否支持音频输出 |

## 与其他模块的关系

- 依赖 `registry.py` 获取前后处理函数和模型能力检查
- 依赖 `executor/abstract.py` 创建执行器实例
- 使用 `request.py` 中的 `OmniDiffusionRequest` 作为输入格式
- 被 `entrypoints/async_omni_diffusion.py` 和 `stage_diffusion_client.py` 调用
- 通过执行器间接与 `worker/` 和 `scheduler.py` 交互

## 总结

`DiffusionEngine` 是扩散模块的顶层入口类，封装了完整的推理流程：前处理 -> 分布式执行 -> 后处理 -> 输出格式化。它通过执行器抽象层支持不同的分布式后端，并内置了 dummy run 预热、性能分析等功能。
