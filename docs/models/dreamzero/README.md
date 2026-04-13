# DreamZero 文档索引

- `docs/models/dreamzero/usage.md`：使用方式、模型目录要求、可配置项、精度/行为边界
- `docs/models/dreamzero/dreamzero.md`：推理调用链与端口设计
- `docs/models/dreamzero/review.md`：实现复审与对齐结论
- `docs/models/dreamzero/todo.md`：剩余事项与精度附录

注意：

- DreamZero root `config.json` 里虽然有
  `action_head_cfg.config.num_inference_timesteps = 4`
- 但当前 upstream eager 与 `vllm-omni` DreamZero 服务链实际对齐的
  denoise 步数口径是 `num_inference_steps = 16`
- 详细说明见 `docs/models/dreamzero/usage.md:307`
