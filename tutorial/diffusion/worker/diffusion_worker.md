# `diffusion_worker.py` — GPU Worker 与进程管理

## 文件概述

`diffusion_worker.py` 包含三个核心类：`DiffusionWorker`（GPU Worker）、`WorkerProc`（进程封装）和 `WorkerWrapperBase`（扩展包装器）。Worker 负责 GPU 设备初始化和分布式环境管理，WorkerProc 运行消息循环处理请求，WorkerWrapperBase 支持通过 mixin 动态扩展 Worker 功能。

## 关键代码解析

### DiffusionWorker — GPU Worker

```python
class DiffusionWorker:
    def __init__(self, local_rank, rank, od_config, skip_load_model=False):
        self.init_device()
        self.model_runner = DiffusionModelRunner(
            vllm_config=self.vllm_config, od_config=self.od_config, device=self.device,
        )
        if not skip_load_model:
            self.load_model(load_format=self.od_config.diffusion_load_format)
            self.init_lora_manager()
```

初始化流程：
1. `init_device()`：设置 CUDA 设备、环境变量、初始化分布式环境（NCCL）和模型并行
2. 创建 `DiffusionModelRunner`
3. 加载模型和 LoRA 管理器

### init_device — 设备与分布式初始化

```python
def init_device(self):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(self.od_config.master_port)
    self.device = current_omni_platform.get_torch_device(rank)

    # 初始化分布式环境
    init_distributed_environment(world_size=world_size, rank=rank)
    initialize_model_parallel(
        data_parallel_size=..., cfg_parallel_size=...,
        sequence_parallel_size=..., tensor_parallel_size=...,
        # HSDP 参数
        fully_shard_degree=..., hsdp_replicate_size=...,
    )
```

### sleep/wake_up — 休眠与唤醒

```python
def sleep(self, level=1):
    if level == 2:
        self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
    allocator = CuMemAllocator.get_instance()
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())

def wake_up(self, tags=None):
    allocator = CuMemAllocator.get_instance()
    allocator.wake_up(tags)
    # 恢复 level 2 保存的 buffers
    if len(self._sleep_saved_buffers):
        for name, buffer in model.named_buffers():
            buffer.data.copy_(self._sleep_saved_buffers[name].data)
```

Level 1 休眠卸载权重，Level 2 额外保存 buffer。

### WorkerProc — 进程封装

```python
class WorkerProc:
    def worker_busy_loop(self):
        while self._running:
            msg = self.recv_message()  # 从 MessageQueue 接收
            if msg.get("type") == "rpc":
                result, should_reply = self.execute_rpc(msg)
                if should_reply:
                    self.return_result(result)
            elif msg.get("type") == "shutdown":
                self._running = False
            else:
                output = self.worker.execute_model(msg, self.od_config)
                self.return_result(output)
```

主循环处理三种消息：RPC 调用、关闭信号和生成请求。只有 Rank 0 创建结果队列并返回结果。

### WorkerWrapperBase — 扩展包装器

```python
class WorkerWrapperBase:
    def _prepare_worker_class(self):
        worker_class = self.base_worker_class
        if self.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(self.worker_extension_cls)
            # 动态创建继承类
            class_name = f"{worker_class.__name__}With{worker_extension_cls.__name__}"
            worker_class = type(class_name, (worker_extension_cls, worker_class), {})
        return worker_class
```

通过 Python 的动态类型创建（`type()`），在运行时将 Worker 扩展类（mixin）与基础 Worker 类合并，实现无侵入式的功能扩展。

### CustomPipelineWorkerExtension — 自定义 Pipeline 扩展

```python
class CustomPipelineWorkerExtension:
    def re_init_pipeline(self, custom_pipeline_args):
        del self.model_runner.pipeline
        gc.collect()
        torch.cuda.empty_cache()
        custom_pipeline_name = custom_pipeline_args["pipeline_class"]
        self.load_model(load_format="custom_pipeline", custom_pipeline_name=custom_pipeline_name)
```

支持在 Worker 初始化后用自定义 Pipeline 替换默认 Pipeline。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionWorker` | 类 | GPU Worker，管理设备、分布式环境、LoRA 和休眠 |
| `WorkerProc` | 类 | 进程封装，运行消息循环 |
| `WorkerWrapperBase` | 类 | Worker 扩展包装器，支持动态 mixin |
| `CustomPipelineWorkerExtension` | 类 | 自定义 Pipeline 替换扩展 |
| `worker_main` | 静态方法 | Worker 进程入口点 |

## 与其他模块的关系

- 创建并使用 `worker/diffusion_model_runner.py` 的 `DiffusionModelRunner`
- 被 `executor/multiproc_executor.py` 通过 `WorkerProc.worker_main` 启动
- 使用 `ipc.py` 的 `pack_diffusion_output_shm` 打包结果
- 使用 `profiler/` 的 `CurrentProfiler` 进行性能分析
- 使用 `distributed/parallel_state.py` 初始化并行环境
- 使用 `lora/manager.py` 的 `DiffusionLoRAManager` 管理 LoRA 适配器

## 总结

`diffusion_worker.py` 实现了完整的 GPU Worker 生命周期管理。`DiffusionWorker` 负责设备和分布式环境的初始化，`WorkerProc` 运行消息处理循环，`WorkerWrapperBase` 通过动态类型创建支持功能扩展。整体架构将基础设施（Worker）与模型操作（Runner）清晰分离。
